from flask import Flask, Response, redirect, request, jsonify, make_response
import os, sys, socket, threading, time, hashlib, secrets
from datetime import datetime, timedelta
import pytz, json
import parquet_cache

# Red de seguridad: si alguna librería de red (gdown/requests) no define su
# propio timeout, esto evita que una conexión colgada bloquee el hilo de
# descarga para siempre.
socket.setdefaulttimeout(160)

# En Render (y en general dentro de contenedores) la salida estándar de Python
# viene "bufferizada por bloque" cuando no hay una terminal real, así que los
# print() de la descarga/arranque pueden tardar minutos en aparecer en los
# logs (o no aparecer hasta que el proceso se reinicia). Forzamos línea por
# línea para que los logs reflejen lo que está pasando en tiempo real.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

def _import_bp(module_name, bp_name):
    """Importa un blueprint de forma segura — si el archivo no existe, avisa pero no tumba la app."""
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return getattr(mod, bp_name)
    except Exception as e:
        print(f"⚠️  No se pudo importar {module_name}.{bp_name}: {e}")
        return None

sesiones_dnt_bp           = _import_bp("sesiones_dnt_flask",           "sesiones_dnt_bp")
familias_sensibilizadas_bp = _import_bp("familias_sensibilizadas_flask", "familias_sensibilizadas_bp")

def _cookie_secure():
    """En Render (HTTPS) las cookies deben tener Secure=True.
    Se activa si la variable de entorno RENDER está presente,
    o si COOKIE_SECURE=true está definida explícitamente.
    En local (sin esas vars) queda False para no romper HTTP.
    """
    import os as _os
    return bool(_os.environ.get("RENDER") or _os.environ.get("COOKIE_SECURE", "").lower() == "true")


# ── DESCARGA DEL PARQUET DESDE GOOGLE DRIVE (solo 2026) ──────────────────────
# Pon aquí el ID del archivo en Google Drive (compartido como "cualquiera con el enlace")
DRIVE_FILE_ID = "189nuCsBwRzN1zacoNIVzf-WwQiC4iNO-"

def _es_parquet_valido(path):
    """Un parquet válido SIEMPRE termina con la firma mágica 'PAR1'.
    Si Google Drive devolvió la página HTML de 'no se puede escanear
    en busca de virus' en lugar del archivo real, esto lo detecta
    antes de que rompa la app (en vez de fallar más tarde con
    'File out of specification: The file must end with PAR1')."""
    try:
        if os.path.getsize(path) < 12:
            return False
        with open(path, "rb") as f:
            f.seek(-4, os.SEEK_END)
            return f.read(4) == b"PAR1"
    except Exception:
        return False

# Guarda el resultado del ÚLTIMO intento de descarga para poder consultarlo
# desde /api/estado-parquet sin tener que rastrear los logs de Render.
_DESCARGA_STATUS = {"ok": None, "mensaje": "aún no se ha intentado descargar", "ts": 0, "tamano_mb": 0}

def _reportar_estado(ok, mensaje, tamano_mb=0):
    _DESCARGA_STATUS["ok"] = ok
    _DESCARGA_STATUS["mensaje"] = mensaje
    _DESCARGA_STATUS["ts"] = time.time()
    _DESCARGA_STATUS["tamano_mb"] = round(tamano_mb, 2)
    print(("✅  " if ok else "❌  ") + mensaje)

def descargar_parquet(forzar=False):
    """Descarga reporte.parquet desde Google Drive usando gdown, que maneja
    automáticamente la página de confirmación 'Google Drive no puede
    escanear este archivo en busca de virus' que aparece en archivos
    grandes (la descarga directa con urllib devolvía esa página HTML en
    vez del parquet real, y eso causaba el error 'must end with PAR1').

    Si ya existe un parquet local válido y reciente, lo reutiliza en vez
    de volver a descargar (evita descargas innecesarias)."""
    destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reporte.parquet")
    os.makedirs(os.path.dirname(destino), exist_ok=True)

    if not DRIVE_FILE_ID or DRIVE_FILE_ID.startswith("PON_AQUI"):
        _reportar_estado(False, "ID de Drive no configurado — saltando descarga")
        return False

    if not forzar and os.path.exists(destino) and _es_parquet_valido(destino):
        antiguedad = time.time() - os.path.getmtime(destino)
        if antiguedad < 1800:
            local_size = os.path.getsize(destino) / 1024 / 1024
            _reportar_estado(True, f"reporte.parquet en caché ({int(antiguedad/60)} min) — sin descargar de nuevo", local_size)
            return True

    try:
        import gdown
    except ImportError:
        _reportar_estado(False, "Falta instalar 'gdown' (revisa que requirements.txt esté actualizado y que Render haya reinstalado dependencias)")
        if os.path.exists(destino) and _es_parquet_valido(destino):
            print("⚠️   Usando reporte.parquet existente en caché")
            return True
        return False

    try:
        tmp = destino + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)

        # Marca "en progreso" ANTES de la llamada bloqueante. Así, si gdown
        # se queda colgado (red lenta/inestable en Render), /api/estado-parquet
        # deja de mostrar para siempre "aún no se ha intentado" y en cambio
        # muestra desde cuándo está atascado — eso ya es un diagnóstico útil
        # aunque la descarga en sí no haya terminado.
        _DESCARGA_STATUS.update({"ok": None, "mensaje": "descarga en progreso...", "ts": time.time(), "tamano_mb": 0})
        print("⬇️   Descargando reporte.parquet desde Google Drive...")

        # Ejecuta gdown en un thread aparte con timeout: si Google Drive no
        # responde o la conexión se cuelga, esto evita quedar bloqueado para
        # siempre sin ningún reporte de error.
        _resultado_dl = {}
        def _hacer_descarga():
            try:
                gdown.download(id=DRIVE_FILE_ID, output=tmp, quiet=False)
                _resultado_dl["ok"] = True
            except Exception as e_dl:
                _resultado_dl["ok"] = False
                _resultado_dl["error"] = e_dl
        t_dl = threading.Thread(target=_hacer_descarga, daemon=True)
        t_dl.start()
        t_dl.join(timeout=150)

        if t_dl.is_alive():
            _reportar_estado(False, "La descarga superó los 150s sin responder (Google Drive no contestó a tiempo) — se reintentará en el próximo ciclo.")
            if os.path.exists(destino) and _es_parquet_valido(destino):
                print("⚠️   Usando reporte.parquet existente en caché")
                return True
            return False

        if _resultado_dl.get("ok") is False:
            raise _resultado_dl["error"]

        if not os.path.exists(tmp) or not _es_parquet_valido(tmp):
            tam = os.path.getsize(tmp) / 1024 / 1024 if os.path.exists(tmp) else 0
            _reportar_estado(False,
                f"El archivo descargado ({tam:.2f} MB) NO es un parquet válido — "
                f"probablemente Drive devolvió una página HTML de permiso/confirmación. "
                f"Revisa que el archivo esté compartido como 'Cualquiera con el enlace'.")
            if os.path.exists(tmp):
                os.remove(tmp)
            if os.path.exists(destino) and _es_parquet_valido(destino):
                print("⚠️   Usando reporte.parquet existente en caché")
                return True
            return False

        os.replace(tmp, destino)
        try:
            _AVANCE_CACHE["ts"] = 0
            _AVANCE_CACHE["data"] = None  # forzar recarga desde cero
            # Borrar caché de disco para que se recalcule con datos nuevos
            try:
                if os.path.exists(_AVANCE_DISK_CACHE):
                    os.remove(_AVANCE_DISK_CACHE)
            except Exception:
                pass
        except NameError:
            pass  # _AVANCE_CACHE aún no definido si se llama al importar
        size_mb = os.path.getsize(destino) / 1024 / 1024
        _reportar_estado(True, f"Descargado correctamente: {size_mb:.1f} MB", size_mb)
        return True
    except Exception as e:
        _reportar_estado(False, f"Error descargando: {e}")
        if os.path.exists(destino) and _es_parquet_valido(destino):
            print("⚠️   Usando reporte.parquet existente en caché")
            return True
        return False

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "rsp-san-pablo-2026-default")
if sesiones_dnt_bp is not None: app.register_blueprint(sesiones_dnt_bp)
if familias_sensibilizadas_bp is not None: app.register_blueprint(familias_sensibilizadas_bp)

# ── SESIONES PERSISTIDAS EN DISCO ────────────────────────────────────────────
# Render puede reiniciar el proceso en cualquier momento y perder la memoria.
# Guardamos las sesiones en un archivo JSON en el mismo directorio del script.
_SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sessions.json")
_sessions_lock = threading.Lock()

def _load_sessions():
    try:
        with _sessions_lock:
            if os.path.exists(_SESSIONS_FILE):
                with open(_SESSIONS_FILE, "r") as _f:
                    return json.load(_f)
    except Exception:
        pass
    return {}

def _save_sessions(s):
    try:
        with _sessions_lock:
            with open(_SESSIONS_FILE, "w") as _f:
                json.dump(s, _f)
    except Exception:
        pass

def _clean_sessions(s):
    now = datetime.now().timestamp()
    return {k: v for k, v in s.items() if v.get("expires", 0) > now}

def create_session(usuario, nombre):
    token = secrets.token_hex(24)
    s = _clean_sessions(_load_sessions())
    s[token] = {
        "usuario": usuario,
        "nombre": nombre,
        "expires": (datetime.now() + timedelta(hours=8)).timestamp()
    }
    _save_sessions(s)
    return token

def get_session(token):
    if not token:
        return None
    s = _load_sessions()
    data = s.get(token)
    if not data:
        return None
    if data.get("expires", 0) < datetime.now().timestamp():
        return None
    return data

def delete_session(token):
    s = _load_sessions()
    s.pop(token, None)
    _save_sessions(s)

_dir = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(_dir, "data", "reporte.parquet")

# Cargar logo al inicio — desde logo_b64.txt o extraído de portada.html
def _load_logo():
    # Opción 1: archivo separado
    p = os.path.join(_dir, "logo_b64.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    # Opción 2: extraer del portada.html (siempre disponible)
    html_path = os.path.join(_dir, "portada.html")
    if os.path.exists(html_path):
        html = open(html_path, encoding="utf-8").read()
        marker = 'src="data:image/png;base64,'
        idx = html.find(marker)
        if idx >= 0:
            start = idx + len(marker)
            end = html.find('"', start)
            return html[start:end]
    return ""

LOGO_B64 = _load_logo()
LIMA_TZ = pytz.timezone("America/Lima")

# ── USUARIOS ─────────────────────────────────────────────────────────────────
# Para cambiar contraseña: genera el hash con:
#   python3 -c "import hashlib; print(hashlib.sha256('TU_CLAVE'.encode()).hexdigest())"
USUARIOS = {
    "admin": {
        "hash": hashlib.sha256("sanpablo2026".encode()).hexdigest(),
        "nombre": "Administrador",
    },
    "cesar": {
        "hash": hashlib.sha256("salud2026".encode()).hexdigest(),
        "nombre": "César E. Malca",
    },
}

# NOTA: la descarga del parquet ya NO se hace aquí de forma bloqueante.
# Se hace en _startup() dentro de un thread background para no bloquear Gunicorn.

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_token():
    return request.cookies.get("rsp_token")

def logged_in():
    return get_session(get_token()) is not None

def current_user():
    data = get_session(get_token())
    return data or {}

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = get_token()
        data  = get_session(token)
        print(f"[AUTH] token={token[:8] if token else 'NONE'}... data={data}")
        if not data:
            resp = make_response(redirect("/login"))
            resp.delete_cookie("rsp_token")
            return resp
        return fn(*args, **kwargs)
    return wrapper

def get_parquet_fecha():
    """Devuelve la fecha máxima del parquet SIN releer el archivo completo.
    Usa parquet_cache.get_fecha_max() que guarda el valor al cargar el df."""
    fecha = parquet_cache.get_fecha_max(2026)
    if fecha:
        return fecha
    import datetime as _dt
    meses = {1:"ENE",2:"FEB",3:"MAR",4:"ABR",5:"MAY",6:"JUN",
             7:"JUL",8:"AGO",9:"SET",10:"OCT",11:"NOV",12:"DIC"}
    try:
        if os.path.exists(PARQUET_PATH):
            mtime = os.path.getmtime(PARQUET_PATH)
            dt_utc = _dt.datetime.fromtimestamp(mtime, tz=pytz.utc)
            dt_lima = dt_utc.astimezone(LIMA_TZ)
            return f"{dt_lima.day:02d} {meses[dt_lima.month]} {dt_lima.year} · {dt_lima.strftime('%H:%M')}"
    except Exception:
        pass
    return None

# ── LOGIN PAGE ────────────────────────────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Red de Salud San Pablo · Acceso</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:'Plus Jakarta Sans',sans-serif;}
:root{--blue:#1C398E;--blue-dark:#0a1a5c;--txt:#0b1e42;--muted:#4a6280;--border:#e2eaf4;--bg:#f0f4fa;}
html,body{height:100%;background:var(--bg);}
body{display:flex;align-items:center;justify-content:center;min-height:100vh;padding:1rem;}

.login-wrap{width:100%;max-width:420px;}

.login-brand{text-align:center;margin-bottom:2rem;}
.brand-logo{width:72px;height:72px;border-radius:50%;object-fit:cover;margin-bottom:1rem;box-shadow:0 4px 20px rgba(28,57,142,.18);}
.brand-title{font-size:1.1rem;font-weight:800;color:var(--txt);letter-spacing:-.01em;}
.brand-sub{font-size:.78rem;color:var(--muted);margin-top:3px;font-weight:500;}

.login-card{background:#fff;border:1px solid var(--border);border-radius:20px;padding:2rem 2.2rem;box-shadow:0 8px 40px rgba(15,40,90,.10);}

.login-head{margin-bottom:1.6rem;}
.login-head h2{font-size:1.3rem;font-weight:800;color:var(--txt);margin-bottom:.3rem;}
.login-head p{font-size:.82rem;color:var(--muted);}

.field{margin-bottom:1.1rem;}
.field label{display:block;font-size:.75rem;font-weight:700;color:var(--txt);letter-spacing:.04em;text-transform:uppercase;margin-bottom:.45rem;}
.field-wrap{position:relative;}
.field input{width:100%;padding:.75rem 1rem .75rem 2.8rem;border:1.5px solid var(--border);border-radius:12px;font-size:.9rem;font-family:inherit;color:var(--txt);background:#fafbff;transition:border .15s,box-shadow .15s;outline:none;}
.field input:focus{border-color:#1C398E;box-shadow:0 0 0 3px rgba(28,57,142,.1);}
.field-ico{position:absolute;left:.85rem;top:50%;transform:translateY(-50%);width:18px;height:18px;opacity:.45;}
.show-pw{position:absolute;right:.85rem;top:50%;transform:translateY(-50%);background:none;border:none;cursor:pointer;padding:0;opacity:.4;transition:opacity .15s;}
.show-pw:hover{opacity:.75;}

.btn-login{width:100%;padding:.85rem;background:linear-gradient(135deg,#1C398E,#2563eb);border:none;border-radius:12px;color:#fff;font-family:inherit;font-size:.92rem;font-weight:700;cursor:pointer;letter-spacing:.02em;transition:all .2s;margin-top:.4rem;}
.btn-login:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(28,57,142,.30);}
.btn-login:active{transform:translateY(0);}

.error{background:#fff0f0;border:1px solid #fecaca;border-radius:10px;padding:.7rem 1rem;font-size:.82rem;color:#dc2626;display:flex;align-items:center;gap:.5rem;margin-bottom:1rem;}
.error svg{flex-shrink:0;}

.login-footer{text-align:center;margin-top:1.5rem;font-size:.72rem;color:#94a3b8;}
</style>
</head>
<body>
<div class="login-wrap">
  <div class="login-brand">
    <img class="brand-logo" src="data:image/png;base64,{LOGO}" alt="Logo Red de Salud San Pablo">
    <div class="brand-title">Red de Salud San Pablo</div>
    <div class="brand-sub">Gobierno Regional de Cajamarca</div>
  </div>

  <div class="login-card">
    <div class="login-head">
      <h2>Iniciar sesión</h2>
      <p>Ingresa tus credenciales para acceder a la plataforma analítica</p>
    </div>

    {ERROR}

    <form method="POST" action="/login">
      <div class="field">
        <label>Usuario</label>
        <div class="field-wrap">
          <svg class="field-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="7" r="4"/><path d="M5 21c0-3.866 3.134-7 7-7s7 3.134 7 7"/></svg>
          <input type="text" name="usuario" placeholder="tu usuario" autocomplete="username" required value="{USR}">
        </div>
      </div>
      <div class="field">
        <label>Contraseña</label>
        <div class="field-wrap">
          <svg class="field-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>
          <input type="password" name="clave" id="pw-field" placeholder="••••••••" autocomplete="current-password" required>
          <button type="button" class="show-pw" onclick="togglePw()" id="pw-btn" aria-label="Mostrar contraseña">
            <svg id="ico-show" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
            <svg id="ico-hide" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
          </button>
        </div>
      </div>
      <button type="submit" class="btn-login">Ingresar a la plataforma</button>
    </form>
  </div>

  <div class="login-footer">
    © 2026 · Red de Salud San Pablo · CEMC
  </div>
</div>
<script>
function togglePw(){
  var f=document.getElementById('pw-field');
  var s=document.getElementById('ico-show');
  var h=document.getElementById('ico-hide');
  if(f.type==='password'){f.type='text';s.style.display='none';h.style.display='';}
  else{f.type='password';s.style.display='';h.style.display='none';}
}
</script>
</body>
</html>"""

# ── RUTAS ─────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET","POST"])
def login():
    logo = LOGO_B64

    error_html = ""
    usr_val = ""

    if request.method == "POST":
        usuario = request.form.get("usuario","").strip().lower()
        clave   = request.form.get("clave","")
        hash_clave = hashlib.sha256(clave.encode()).hexdigest()
        usr_val = usuario

        if usuario in USUARIOS and USUARIOS[usuario]["hash"] == hash_clave:
            token = create_session(usuario, USUARIOS[usuario]["nombre"])
            resp = make_response(redirect("/"))
            resp.set_cookie("rsp_token", token,
                           max_age=28800,  # 8 horas
                           httponly=True,
                           secure=_cookie_secure(),
                           samesite="Lax",
                           path="/")
            return resp
        else:
            error_html = """<div class="error">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              Usuario o contraseña incorrectos. Intenta de nuevo.
            </div>"""

    html = LOGIN_HTML.replace("{LOGO}", logo).replace("{ERROR}", error_html).replace("{USR}", usr_val)
    return Response(html, mimetype="text/html; charset=utf-8")

@app.route("/home")
def home():
    """Regreso desde un módulo — redirige a portada si hay sesión válida en cookie."""
    if logged_in():
        return redirect("/")
    # No hay cookie válida → login
    resp = make_response(redirect("/login"))
    resp.delete_cookie("rsp_token")
    return resp

@app.route("/logout")
def logout():
    delete_session(get_token())
    resp = make_response(redirect("/login"))
    resp.delete_cookie("rsp_token")
    return resp

@app.before_request
def _antes_de_request():
    global _REQUESTS_ACTIVAS
    with _REQUESTS_LOCK:
        _REQUESTS_ACTIVAS += 1

@app.teardown_request
def _despues_de_request(exc):
    global _REQUESTS_ACTIVAS
    with _REQUESTS_LOCK:
        _REQUESTS_ACTIVAS = max(0, _REQUESTS_ACTIVAS - 1)

@app.route("/")
@login_required
def portada():
    with open(os.path.join(_dir, "portada.html"), encoding="utf-8") as f:
        content = f.read()
    # Inyectar nombre del usuario y botón logout en el navbar
    nombre = current_user().get("nombre","")
    user_badge = f"""<div class="nd-user-badge">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="7" r="4"/><path d="M5 21c0-3.866 3.134-7 7-7s7 3.134 7 7"/></svg>
      {nombre}
    </div>
    <a href="/logout" class="nd-logout-btn">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
      Salir
    </a>"""
    # Agregar CSS del badge/logout y el HTML antes del cierre del nav-right
    extra_css = """
.nd-user-badge{display:flex;align-items:center;gap:5px;font-size:.65rem;font-weight:700;color:var(--blue);letter-spacing:.03em;background:#f0f4ff;border:1px solid #d0dcff;border-radius:20px;padding:4px 12px;white-space:nowrap;}
.nd-logout-btn{display:flex;align-items:center;gap:5px;font-size:.65rem;font-weight:700;color:#e05;text-decoration:none;background:#fff0f3;border:1px solid #ffd0db;border-radius:20px;padding:4px 12px;transition:background .15s;white-space:nowrap;}
.nd-logout-btn:hover{background:#ffe0e8;}
"""
    content = content.replace("</style>", extra_css + "</style>", 1)
    content = content.replace(
        '<div class="nd-dot"></div>',
        user_badge + '<div class="nd-dot"></div>'
    )
    return Response(content, mimetype="text/html; charset=utf-8")

@app.route("/api/update-time")
@login_required
def api_update_time():
    fecha = get_parquet_fecha()
    if fecha:
        return jsonify({"fecha": fecha, "ok": True})
    dt_lima = datetime.now(LIMA_TZ)
    meses = {1:"ENE",2:"FEB",3:"MAR",4:"ABR",5:"MAY",6:"JUN",
             7:"JUL",8:"AGO",9:"SET",10:"OCT",11:"NOV",12:"DIC"}
    fecha_now = f"{dt_lima.day:02d} {meses[dt_lima.month]} {dt_lima.year} · {dt_lima.strftime('%H:%M')}"
    return jsonify({"fecha": fecha_now, "ok": False})


@app.route("/health")
def health():
    """Health check sin login — Render lo usa para saber si el worker está vivo.
    Responde 200 inmediatamente aunque los datos aún estén descargando."""
    import parquet_cache as _pc
    datos_ok = bool(_pc._CACHE)
    return jsonify({"status": "ok", "datos": datos_ok}), 200


@app.route("/api/estado-parquet")
@login_required
def api_estado_parquet():
    """Diagnóstico rápido del parquet sin tener que leer los logs de Render:
    visita esta URL logueado para ver si la descarga funcionó, cuándo, qué
    tamaño quedó y el último error (si lo hubo)."""
    existe = os.path.exists(PARQUET_PATH)
    valido = _es_parquet_valido(PARQUET_PATH) if existe else False
    tam_mb = round(os.path.getsize(PARQUET_PATH) / 1024 / 1024, 2) if existe else 0
    ultima = dict(_DESCARGA_STATUS)
    if ultima.get("ts"):
        ultima["hace"] = f"{int((time.time() - ultima['ts']) / 60)} min"
    return jsonify({
        "drive_file_id": DRIVE_FILE_ID,
        "archivo_existe": existe,
        "archivo_valido_par1": valido,
        "tamano_actual_mb": tam_mb,
        "ultimo_intento_descarga": ultima,
    })


# ── CACHÉ Y CÁLCULO DE AVANCE DE METAS (solo los 2 módulos de este proyecto) ─
# El caché de avance se persiste en disco para sobrevivir reinicios del worker.
# Solo se recalcula cuando el reporte.parquet cambia (tamaño o mtime distintos).
_AVANCE_CACHE      = {"data": None, "ts": 0}
_AVANCE_CALCULANDO = False  # evita doble cálculo concurrente
_REQUESTS_ACTIVAS  = 0      # contador de requests en curso
_REQUESTS_LOCK     = threading.Lock()

_AVANCE_DISK_CACHE = os.path.join(_dir, "avance_cache.json")

def _parquet_signature():
    """Devuelve (mtime, size) del reporte.parquet principal.
    Sirve como firma para detectar si cambió."""
    try:
        st = os.stat(PARQUET_PATH)
        return round(st.st_mtime, 2), st.st_size
    except Exception:
        return 0, 0

def _load_avance_disk():
    """Carga el caché de avance desde disco. Devuelve None si no existe o es inválido."""
    try:
        if not os.path.exists(_AVANCE_DISK_CACHE):
            return None
        with open(_AVANCE_DISK_CACHE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # Verificar que el parquet no cambió desde que se guardó el caché
        sig_guardada = tuple(saved.get("parquet_sig", [0, 0]))
        sig_actual   = _parquet_signature()
        if sig_guardada != sig_actual:
            print(f"[avance_cache] parquet cambió → recalcular (guardado={sig_guardada} actual={sig_actual})")
            return None
        data = saved.get("data")
        if not data:
            return None
        print("[avance_cache] ✅ Cargado desde disco (parquet sin cambios)")
        return data
    except Exception as e:
        print(f"[avance_cache] Error leyendo disco: {e}")
        return None

def _save_avance_disk(data):
    """Guarda el resultado del avance en disco junto con la firma del parquet."""
    try:
        sig = list(_parquet_signature())
        payload = {"data": data, "parquet_sig": sig}
        tmp = _AVANCE_DISK_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, _AVANCE_DISK_CACHE)
        print(f"[avance_cache] 💾 Guardado en disco (sig={sig})")
    except Exception as e:
        print(f"[avance_cache] Error guardando disco: {e}")

def _calcular_avance_real():
    """Calcula el avance de los 2 módulos de este proyecto (sesiones_dnt y
    familias_sensibilizadas), con pausa entre cada uno.

    Corre en un daemon thread. La pausa entre módulos evita que Polars/Pandas
    monopolicen el GIL y bloqueen las peticiones HTTP del worker gthread
    durante el cálculo inicial.
    """
    import time, importlib, gc
    t0 = time.time()
    MODULOS_CFG = [
        ("sesiones_dnt_flask",           "sesiones_dnt"),
        ("familias_sensibilizadas_flask", "familias_sensibilizadas"),
    ]
    resultado = dict(_AVANCE_CACHE.get("data") or {})  # preservar valores ya calculados
    for mod_name, key in MODULOS_CFG:
        try:
            mod = importlib.import_module(mod_name)
            ret = mod.procesar_datos(ipress_sel=[], mes_sel=[], dni_raw="")
            df_final = ret[0]
            if df_final is None or (hasattr(df_final, "empty") and df_final.empty):
                resultado[key] = 0.0
            elif "Avance %" in df_final.columns:
                resultado[key] = round(float(df_final["Avance %"].mean()), 1)
            else:
                resultado[key] = 0.0
            print(f"  [avance:{key}] {resultado[key]}%")
            del df_final, ret
            # Publicar resultado parcial inmediatamente — el JS lo verá antes
            # de que termine el segundo módulo
            _AVANCE_CACHE["data"] = dict(resultado)
            _AVANCE_CACHE["ts"]   = time.time()
            gc.collect()
            time.sleep(2)   # ceder el GIL para que Flask pueda atender peticiones
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [avance:{key}] ERROR: {e}")
            resultado[key] = None
            gc.collect()
            time.sleep(1)
    _AVANCE_CACHE["data"] = resultado
    _AVANCE_CACHE["ts"]   = time.time()
    _save_avance_disk(resultado)
    print(f"[avance] completado en {time.time()-t0:.1f}s → {resultado}")

def calcular_avance_metas():
    """Devuelve el avance desde caché. Nunca bloquea al usuario.
    Orden de prioridad:
      1. Caché en memoria (mismo proceso, ts fresco)
      2. Caché en disco   (sobrevive reinicios; válido si parquet no cambió)
      3. Recálculo en background (devuelve {} al usuario mientras tanto)
    """
    global _AVANCE_CALCULANDO
    import time
    now = time.time()

    # 1. Caché en memoria fresco (calculado en esta sesión)
    if _AVANCE_CACHE["data"] is not None and _AVANCE_CACHE["ts"] > 0:
        return _AVANCE_CACHE["data"]

    # 2. Intentar cargar desde disco (sobrevive reinicios del worker)
    if _AVANCE_CACHE["data"] is None:
        data_disco = _load_avance_disk()
        if data_disco:
            _AVANCE_CACHE["data"] = data_disco
            _AVANCE_CACHE["ts"]   = now
            return data_disco

    # 3. Sin caché válido: devolver lo que haya (puede ser {}) y recalcular en background
    if not _AVANCE_CALCULANDO:
        def _primer_calculo():
            global _AVANCE_CALCULANDO
            _AVANCE_CALCULANDO = True
            try:
                _calcular_avance_real()
            finally:
                _AVANCE_CALCULANDO = False
        threading.Thread(target=_primer_calculo, daemon=True).start()
    return _AVANCE_CACHE["data"] or {}


@app.route("/api/avance")
@login_required
def api_avance():
    avance = calcular_avance_metas()
    fecha  = get_parquet_fecha()
    # "calculando": True si no hay datos aún — el JS reintentará en 3s
    tiene_datos = bool(avance) and any(v is not None for v in avance.values())
    calculando  = not tiene_datos
    return jsonify({"avance": avance, "fecha": fecha or "", "ok": tiene_datos, "calculando": calculando})

@app.route("/api/refresh")
@login_required
def api_refresh():
    """Fuerza la descarga del parquet desde Drive e invalida los cachés de los 2 módulos."""
    try:
        from sesiones_dnt_flask import invalidar_cache as _ic_sd;     _ic_sd()
    except Exception: pass
    try:
        from familias_sensibilizadas_flask import invalidar_cache as _ic_fs; _ic_fs()
    except Exception: pass
    try:
        parquet_cache.invalidar()
    except Exception: pass

    ok = descargar_parquet(forzar=True)
    avance = calcular_avance_metas()
    fecha  = get_parquet_fecha()
    return jsonify({"ok": ok, "fecha": fecha or "", "avance": avance,
                    "mensaje": "Datos actualizados desde Google Drive" if ok else "Error al actualizar"})

# ── 404 personalizado ─────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    html = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Página no encontrada</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;700;900&display=swap" rel="stylesheet">
<style>*{margin:0;padding:0;box-sizing:border-box;font-family:'Plus Jakarta Sans',sans-serif;}
body{min-height:100vh;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0a1a5c,#1C398E);}
.box{text-align:center;color:#fff;padding:2rem;}
.num{font-size:6rem;font-weight:900;opacity:.2;line-height:1;}
h2{font-size:1.5rem;font-weight:800;margin-bottom:.5rem;}
p{font-size:.9rem;color:rgba(255,255,255,.7);margin-bottom:2rem;}
a{color:#fff;text-decoration:none;border:1px solid rgba(255,255,255,.4);padding:8px 24px;border-radius:20px;font-size:.85rem;font-weight:600;}
a:hover{background:rgba(255,255,255,.1);}
</style></head>
<body><div class="box">
<div class="num">404</div>
<h2>Página no encontrada</h2>
<p>La ruta que buscas no existe en esta plataforma.</p>
<a href="/">← Volver a la portada</a>
</div></body></html>"""
    return Response(html, status=404, mimetype="text/html; charset=utf-8")

# ── INICIO ────────────────────────────────────────────────────────────────────
def precalentar_modulos():
    """Carga el DataFrame 2026 compartido — ambos módulos lo reutilizan."""
    import gc
    try:
        t0 = time.time()
        df, err = parquet_cache.get_df(2026)
        if err:
            print(f"   ⚠️ parquet_cache: {err}")
        else:
            print(f"   🔥 parquet 2026 listo en {time.time()-t0:.1f}s")
        gc.collect()
    except Exception as e:
        print(f"   ⚠️ precalentar_modulos: {e}")
    print("   ✅ Caché compartido listo")


# ── STARTUP (corre tanto con Gunicorn en Render como con python local) ────────
def _startup():
    """Inicialización al arrancar.

    TODO ocurre en un thread background para no bloquear el worker de Gunicorn.
    Render mata el worker si tarda más de 30 s en responder la primera petición.
    Con este esquema Flask responde inmediatamente; los datos llegan en ~30-60 s.
    """
    try:
        if os.path.exists(_SESSIONS_FILE):
            os.remove(_SESSIONS_FILE)
    except Exception:
        pass

    def _bg():
        # 1. Descargar reporte.parquet (2026)
        try:
            descargar_parquet()
        except Exception as e:
            print(f"[startup] descarga parquet: {e}")
        # 2. Cargar en caché compartido
        try:
            precalentar_modulos()
        except Exception as e:
            print(f"[startup] precalentar: {e}")
        # 3. Intentar restaurar avance desde disco (evita recalcular si el parquet no cambió)
        data_disco = _load_avance_disk()
        if data_disco:
            _AVANCE_CACHE["data"] = data_disco
            _AVANCE_CACHE["ts"]   = time.time()
            print("[startup] ✅ Avance restaurado desde disco — sin recálculo necesario")
        else:
            # Sin caché válido en disco: esperar unos segundos y recalcular
            print("[startup] 🔄 No hay caché de avance en disco → recalculando en 5s...")
            time.sleep(5)
            try:
                _calcular_avance_real()
            except Exception as e:
                print(f"[startup] avance: {e}")

    threading.Thread(target=_bg, daemon=True).start()

# Ejecutar al importar el módulo — funciona con Gunicorn y con python directo
_startup()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print("\n🚀  Red de Salud San Pablo - Seguimiento Nutricional")
    print("=" * 50)
    print(f"✅  Portada  → http://localhost:{port}")
    print(f"🔐  Login    → http://localhost:{port}/login")
    print()
    print("👤  Usuarios configurados:")
    for u, d in USUARIOS.items():
        print(f"    · {u} ({d['nombre']})")
    fecha = get_parquet_fecha()
    if fecha:
        print(f"\n📊  reporte.parquet: {fecha}")
    print("=" * 50)

    descargar_parquet()

    app.run(host="0.0.0.0", port=port, debug=False)
