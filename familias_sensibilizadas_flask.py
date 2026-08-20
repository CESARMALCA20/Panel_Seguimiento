"""
Ruta Flask para Familias Sensibilizadas
Agrega esto a Portada.py:
    from familias_sensibilizadas_flask import familias_sensibilizadas_bp
    app.register_blueprint(familias_sensibilizadas_bp)
"""

from flask import Blueprint, Response, request
import parquet_cache
import os, re, json
import polars as pl
import pandas as pd
from datetime import datetime

familias_sensibilizadas_bp = Blueprint("familias_sensibilizadas", __name__)

_dir = os.path.dirname(os.path.abspath(__file__))

PARQUET_PATH = os.path.join(_dir, "data", "reporte.parquet")

# ── ITEMS CONFIG ──────────────────────────────────────────────────────────────
# C0009: filtra Valor_Lab == TA | AD | HA | SSI
# C2062: filtra Valor_Lab == "24"

ITEMS_CONFIG = [
    # Seccion TA  — 1° Visita
    ("C0009", "C0009_TA",    "C0009"),
    ("C2062", "C2062_24_TA", "C2062"),
    # Seccion AD  — 2° Visita
    ("C0009", "C0009_AD",    "C0009"),
    ("C2062", "C2062_24_AD", "C2062"),
]
N_ITEMS = len(ITEMS_CONFIG)

# Grupos para la doble cabecera de la tabla (titulo, n_cols, color_hex, [id_cols])
GRUPOS = [
    ("SESIONES EDUCATIVAS A FAMILIAS (Familia Capacitada): 1° VISITA", 2, "#1e40af", ["C0009_TA", "C2062_24_TA"]),
    ("SESIONES EDUCATIVAS A FAMILIAS (Familia Capacitada): 2° VISITA", 2, "#065f46", ["C0009_AD", "C2062_24_AD"]),
]

SOLO_MUJER  = set()
SOLO_HOMBRE = set()

MESES_ES = {
    "January":"Enero","February":"Febrero","March":"Marzo","April":"Abril",
    "May":"Mayo","June":"Junio","July":"Julio","August":"Agosto",
    "September":"Septiembre","October":"Octubre","November":"Noviembre","December":"Diciembre"
}

# Mapa: seccion -> Valor_Lab de C0009 que la identifica
_SECCION_VLAB = {"TA": "TA", "AD": "AD", "HA": "HA", "SSI": "SSI"}


def _extraer_visitas_c0009(df: pl.DataFrame) -> tuple[dict, dict]:
    """
    Nueva lógica: agrupa C0009 por paciente y fecha.
    - Por cada paciente obtiene todas las fechas únicas ordenadas ASC.
    - 1° visita = fecha más antigua; 2° visita = segunda fecha distinta.
    - Si en una misma fecha hay varios Valor_Lab (TA, AD, HA, SSI)
      los concatena: "dd/mm/yyyy (TA·AD·SSI)".
    - C2062 (Valor_Lab=24) se busca en esa misma fecha exacta.
    Retorna:
        visita1: {dni: "dd/mm/yyyy (VLABS)"}
        visita2: {dni: "dd/mm/yyyy (VLABS)"}
    """
    VLABS_C0009 = {"TA", "AD", "HA", "SSI"}

    sub = df.filter(
        (pl.col("Codigo_Item") == "C0009") &
        (pl.col("Valor_Lab").str.to_uppercase().is_in(VLABS_C0009))
    )
    if sub.is_empty():
        return {}, {}

    # Agrupar por DNI + Fecha → lista de vlabs en esa fecha
    grp = (
        sub
        .with_columns(pl.col("Valor_Lab").str.to_uppercase().alias("vlab_up"))
        .group_by(["Numero_Documento_Paciente", "Fecha_Atencion"])
        .agg(pl.col("vlab_up").unique().sort().alias("vlabs"))
        .sort(["Numero_Documento_Paciente", "Fecha_Atencion"])
    )

    visita1, visita2 = {}, {}
    for row in grp.iter_rows(named=True):
        dni   = str(row["Numero_Documento_Paciente"])
        fecha = row["Fecha_Atencion"]
        if fecha is None:
            continue
        fecha_str = fecha.strftime("%d/%m/%Y")
        vlabs_str = "·".join(row["vlabs"])
        celda     = f"{fecha_str} ({vlabs_str})"

        if dni not in visita1:
            visita1[dni] = celda          # primera fecha vista → visita 1
        elif dni not in visita2:
            visita2[dni] = celda          # segunda fecha distinta → visita 2

    return visita1, visita2


def _extraer_c2062_para_visita(df: pl.DataFrame, fechas_visita: dict) -> dict:
    """
    Busca C2062 (Valor_Lab=24) en la misma fecha que la visita dada.
    Si en esa fecha hay varios Valor_Lab los concatena igual que C0009.
    fechas_visita: {dni: "dd/mm/yyyy (VLABS)"}  — resultado de _extraer_visitas_c0009
    Retorna: {dni: "dd/mm/yyyy (24)"}
    """
    sub = df.filter(
        (pl.col("Codigo_Item") == "C2062") &
        (pl.col("Valor_Lab").str.to_uppercase() == "24")
    )
    if sub.is_empty():
        return {}

    # Mapa dni -> set de fechas donde existe C2062/24
    grp = (
        sub
        .group_by("Numero_Documento_Paciente")
        .agg(pl.col("Fecha_Atencion").alias("fechas"))
    )
    fechas_c2062 = {}
    for row in grp.iter_rows(named=True):
        dni = str(row["Numero_Documento_Paciente"])
        fechas_c2062[dni] = {
            f.strftime("%d/%m/%Y") for f in row["fechas"] if f is not None
        }

    resultado = {}
    for dni, celda_c0009 in fechas_visita.items():
        fecha_visita = celda_c0009[:10]   # "dd/mm/yyyy"
        if fecha_visita in fechas_c2062.get(dni, set()):
            resultado[dni] = f"{fecha_visita} (24)"
    return resultado


_DF_FILTRADO = {}  # {anio: DataFrame}

def invalidar_cache():
    """Llama esto al actualizar el parquet para forzar recarga del DataFrame."""
    global _DF_FILTRADO
    _DF_FILTRADO = {}

def _cast_fecha(col_name, df):
    """Castea columna a pl.Date si viene como i32/i64/Datetime desde parquet."""
    if col_name not in df.columns:
        return df
    dtype = df[col_name].dtype
    if dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Datetime):
        return df.with_columns(pl.col(col_name).cast(pl.Date))
    return df

def _get_df():
    """Obtiene el DataFrame con C0009/C2062 desde el caché compartido."""
    global _DF_FILTRADO
    if "2026" in _DF_FILTRADO:
        return _DF_FILTRADO["2026"], None
    df_raw, err = parquet_cache.get_df()
    if err:
        return None, err
    try:
        df = df_raw.filter(
            pl.col("Codigo_Item").is_in(["C0009", "C2062"])
        )
        _DF_FILTRADO["2026"] = df
        return df, None
    except Exception as e:
        return None, str(e)


def procesar_datos(ipress_sel, mes_sel, dni_raw, anio=2026):
    try:
        df_raw, err = _get_df()
        if err:
            return None, err, [], []

        df = df_raw

        lista_ipress = sorted([str(i).strip() for i in df["Nombre_Establecimiento"].unique().to_list()])
        lista_meses_df = df.select(["Mes_Num", "Mes_Nombre"]).unique().sort("Mes_Num")
        lista_meses = lista_meses_df["Mes_Nombre"].to_list()

        lista_dni = re.findall(r'\d+', dni_raw) if dni_raw else []
        if lista_dni:
            df = df.filter(pl.col("Numero_Documento_Paciente").cast(pl.Utf8).str.strip_chars().is_in(lista_dni))
        else:
            if ipress_sel:
                df = df.filter(pl.col("Nombre_Establecimiento").is_in(ipress_sel))
            if mes_sel:
                df = df.filter(pl.col("Mes_Nombre").is_in(mes_sel))

        if df.is_empty():
            return None, None, lista_ipress, lista_meses

        # ── Num_Pag y Num_Reg: del C0009 más antiguo (1° visita) ──
        def _extraer_num_map(df_src):
            """Retorna {dni: (Num_Pag, Num_Reg)} del C0009 más antiguo."""
            VLABS_C0009 = {"TA", "AD", "HA", "SSI"}
            sub = df_src.filter(
                (pl.col("Codigo_Item") == "C0009") &
                (pl.col("Valor_Lab").cast(pl.Utf8).str.to_uppercase().is_in(VLABS_C0009))
            ).filter(
                pl.col("Numero_Documento_Paciente").cast(pl.Utf8).str.strip_chars().str.len_chars() > 0
            ).sort("Fecha_Atencion", descending=False)
            if sub.is_empty():
                return {}
            _cols = set(sub.columns)
            agg = []
            if "Num_Pag" in _cols:
                agg.append(pl.col("Num_Pag").cast(pl.Utf8).fill_null("").first().alias("Num_Pag"))
            if "Num_Reg" in _cols:
                agg.append(pl.col("Num_Reg").cast(pl.Utf8).fill_null("").first().alias("Num_Reg"))
            if not agg:
                return {}
            grp = sub.group_by("Numero_Documento_Paciente").agg(agg)
            result = {}
            for r in grp.iter_rows(named=True):
                result[str(r["Numero_Documento_Paciente"])] = (
                    r.get("Num_Pag", ""), r.get("Num_Reg", "")
                )
            return result

        _num_map = _extraer_num_map(df)

        def _get_num(dni):
            return _num_map.get(dni, ("", ""))

        # ── Lote: el más frecuente entre todos los registros C0009/C2062 ──
        # ── Demás campos: del registro C0009/C2062 más reciente ──
        df_sorted = df.filter(
            pl.col("Numero_Documento_Paciente").cast(pl.Utf8).str.strip_chars().str.len_chars() > 0
        ).sort("Fecha_Atencion", descending=True)

        agg_exprs = [
            pl.col("Fecha_Atencion").first(),
            pl.col("Nombre_Establecimiento").first(),
            pl.concat_str([
                pl.col("Apellido_Paterno_Paciente").fill_null(""),
                pl.col("Apellido_Materno_Paciente").fill_null(""),
                pl.col("Nombres_Paciente").fill_null(""),
            ], separator=" ").first().alias("Paciente"),
            pl.col("Fecha_Nacimiento_Paciente").first(),
            pl.col("Anio_Actual_Paciente").first(),
            pl.col("Genero").first(),
        ]
        if "Lote" in df_sorted.columns:
            agg_exprs.append(
                pl.col("Lote").cast(pl.Utf8).fill_null("")
                .value_counts(sort=True).struct.field("Lote").first()
                .alias("Lote")
            )
        personal_cols = ["Apellido_Paterno_Personal", "Apellido_Materno_Personal", "Nombres_Personal"]
        if all(c in df_sorted.columns for c in personal_cols):
            agg_exprs.append(
                pl.concat_str([
                    pl.col("Apellido_Paterno_Personal").cast(pl.Utf8).fill_null(""),
                    pl.col("Apellido_Materno_Personal").cast(pl.Utf8).fill_null(""),
                    pl.col("Nombres_Personal").cast(pl.Utf8).fill_null(""),
                ], separator=" ").first().str.strip_chars().alias("Nombre_Personal")
            )

        df_info = (
            df_sorted
            .group_by("Numero_Documento_Paciente")
            .agg(agg_exprs)
            .sort("Fecha_Atencion", descending=True)
        )

        ids_pivot = [c[1] for c in ITEMS_CONFIG]

        # ── Nueva lógica: visitas por orden de fecha ──────────────────────────
        # 1° visita = fecha más antigua de C0009; 2° visita = segunda fecha distinta
        # Si en una misma fecha hay varios Valor_Lab → se concatenan con ·
        c0009_v1, c0009_v2 = _extraer_visitas_c0009(df)
        c2062_v1 = _extraer_c2062_para_visita(df, c0009_v1)
        c2062_v2 = _extraer_c2062_para_visita(df, c0009_v2)

        resultado_items = {
            "C0009_TA":    c0009_v1,
            "C2062_24_TA": c2062_v1,
            "C0009_AD":    c0009_v2,
            "C2062_24_AD": c2062_v2,
        }

        df_final = df_info.to_pandas()
        df_final["Numero_Documento_Paciente"] = df_final["Numero_Documento_Paciente"].astype(str)

        # Asignar Num_Pag y Num_Reg
        _tiene_num = bool(_num_map)
        if _tiene_num:
            df_final["Num_Pag"] = df_final["Numero_Documento_Paciente"].apply(lambda d: _get_num(d)[0])
            df_final["Num_Reg"] = df_final["Numero_Documento_Paciente"].apply(lambda d: _get_num(d)[1])

        for id_col in ids_pivot:
            mapa = resultado_items.get(id_col, {})
            df_final[id_col] = df_final["Numero_Documento_Paciente"].map(mapa)

        # NO renombramos — las columnas de indicadores usan el id_col directamente
        columnas_indicadores = ids_pivot

        def _es_realizado(val):
            if val is None or (isinstance(val, float) and pd.isna(val)): return False
            if isinstance(val, str) and val in ["❌", "NO APLICA"]: return False
            return True

        df_final["Realizados"] = df_final[columnas_indicadores].apply(
            lambda col: col.map(_es_realizado)
        ).sum(axis=1)

        df_final["_n"]       = N_ITEMS
        df_final["Faltan"]   = df_final["_n"] - df_final["Realizados"]
        df_final["Avance %"] = ((df_final["Realizados"] / df_final["_n"]) * 100).round(1)
        df_final = df_final.drop(columns=["_n"])
        df_final[columnas_indicadores] = df_final[columnas_indicadores].fillna("❌")

        # Solo mostrar pacientes con AL MENOS un C0009/C2062 realizado
        df_final = df_final[df_final["Realizados"] > 0].copy()

        df_final = df_final.sort_values("Avance %", ascending=False).reset_index(drop=True)
        df_final.index = df_final.index + 1

        return df_final, columnas_indicadores, lista_ipress, lista_meses

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e), [], []


def _get_parquet_fecha():
    """Usa parquet_cache.get_fecha_max() — SIN releer el archivo."""
    return parquet_cache.get_fecha_max(2026)


def render_page(df_final, columnas_indicadores, lista_ipress, lista_meses,
                ipress_sel, mes_sel, dni_raw, error=None, aviso=None):

    _fecha_badge = _get_parquet_fecha()

    # ── Filas de la tabla ──────────────────────────────────────────────────────
    rows_html = ""
    if df_final is not None and not df_final.empty:
        for i, (_, row) in enumerate(df_final.iterrows(), 1):
            av  = float(row.get("Avance %", 0))
            bg  = "#f0fdf4" if av >= 99.9 else "#fefce8" if av >= 50 else "#fff1f2"
            sbg = f"background:{bg};"

            pend = [col for col in columnas_indicadores if str(row.get(col, "")).strip() == "❌"]
            pend_html = "".join(
                f'<div class="tip-item">'
                f'<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.5" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>'
                f'{p}</div>'
                for p in pend
            ) if pend else '<div class="tip-ok">✅ Todos completados</div>'

            _pd = pend_html.replace('"', '&quot;').replace("'", "&#39;")
            _pac_nombre = str(row.get("Paciente", "")).strip().replace('"', '&quot;')
            eye_btn = (
                f'<button class="eye-btn" data-tip="{_pd}" data-pac="{_pac_nombre}" '
                f'onclick="togglePostit(this)">' +
                '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>'
                '<circle cx="12" cy="12" r="3"/></svg></button>'
            )

            celdas = (
                f'<td class="idx s-idx" style="{sbg}">{i}</td>'
                f'<td class="s-dni" style="{sbg}">{row.get("Numero_Documento_Paciente","")}</td>'
                f'<td class="s-pac" style="{sbg}">{row.get("Paciente","")} {eye_btn}</td>'
            )

            _opc = [c for c in ["Num_Pag", "Num_Reg"] if c in row.index]
            rest_cols = [
                "Fecha_Nacimiento_Paciente",
                "Nombre_Establecimiento", "Anio_Actual_Paciente", "Genero",
                "Nombre_Personal", "Lote",
            ] + _opc + ["Realizados", "Faltan", "Avance %"] + columnas_indicadores

            for col in rest_cols:
                val = row.get(col, "")
                if isinstance(val, (pd.Timestamp, datetime)):
                    disp = "" if pd.isnull(val) else val.strftime("%d/%m/%Y")
                elif col == "Anio_Actual_Paciente":
                    try: disp = str(int(float(val))) if val != "" and not (isinstance(val, float) and pd.isna(val)) else ""
                    except: disp = str(val)
                elif col == "Avance %" and val != "":
                    disp = f"{float(val):.1f}"
                else:
                    disp = str(val) if not (isinstance(val, float) and pd.isna(val)) else ""

                if col in columnas_indicadores:
                    if disp == "❌":          css = "no"
                    elif disp == "NO APLICA": css = "na"
                    elif disp.startswith("⚠"): css = "warn"
                    else:                      css = "ok"
                elif col == "Avance %":
                    try: v=float(val); css = "avok" if v>=99.9 else "avmd" if v>=50 else "avno"
                    except: css = ""
                elif col == "Realizados": css = "num"
                elif col == "Faltan":     css = "grey"
                else: css = ""
                celdas += f'<td class="{css}">{disp}</td>'

            rows_html += f"<tr>{celdas}</tr>"

    # ── Etiquetas de filtros ───────────────────────────────────────────────────
    if not ipress_sel:
        ipress_label = "Todas las IPRESS"
    elif len(ipress_sel) == 1:
        ipress_label = ipress_sel[0]
    else:
        ipress_label = f"{len(ipress_sel)} IPRESS seleccionadas"

    if not mes_sel:
        mes_label = "Todos los meses"
    elif len(mes_sel) == 1:
        mes_label = mes_sel[0]
    else:
        mes_label = f"{len(mes_sel)} meses seleccionados"

    lista_ipress = [i for i in lista_ipress if i and str(i).strip()]
    ipress_opts = "".join(
        f'<label class="ms-opt">'
        f'<input type="checkbox" name="ipress" value="{i}" {"checked" if i in ipress_sel else ""}>'
        f'<span>{i}</span></label>'
        for i in lista_ipress
    )
    mes_opts = "".join(
        f'<label class="ms-opt">'
        f'<input type="checkbox" name="mes" value="{m}" {"checked" if m in mes_sel else ""}>'
        f'<span>{m}</span></label>'
        for m in lista_meses
    )

    # ── Métricas ───────────────────────────────────────────────────────────────
    metrics_html = ""
    if df_final is not None and not df_final.empty:
        total     = len(df_final)
        completos = len(df_final[df_final["Avance %"] >= 99.9])
        en_proc   = len(df_final[(df_final["Avance %"] >= 50) & (df_final["Avance %"] < 99.9)])
        criticos  = len(df_final[df_final["Avance %"] < 50])
        prom      = df_final["Avance %"].mean()
        col_prom  = "#059669" if prom >= 70 else "#b45309" if prom >= 40 else "#dc2626"
        metrics_html = f"""
        <div class="metrics">
          <div class="mc"><div class="ml">Total Personas</div><div class="mv">{total}</div></div>
          <div class="mc"><div class="ml">Completos 100%</div><div class="mv" style="color:#059669">{completos}</div></div>
          <div class="mc"><div class="ml">En Proceso ≥50%</div><div class="mv" style="color:#b45309">{en_proc}</div></div>
          <div class="mc"><div class="ml">Críticos &lt;50%</div><div class="mv" style="color:#dc2626">{criticos}</div></div>
          <div class="mc"><div class="ml">Avance Promedio</div><div class="mv" style="color:{col_prom}">{prom:.1f}%</div></div>
        </div>"""

    # ── Encabezados de tabla con doble fila (grupos) ──────────────────────────
    _opc_hdrs = []
    if df_final is not None and not df_final.empty:
        if "Num_Pag" in df_final.columns: _opc_hdrs.append("Num_Pag")
        if "Num_Reg" in df_final.columns: _opc_hdrs.append("Num_Reg")
    fixed_headers = ["Fecha Nacimiento Pac.", "IPRESS", "Edad",
                     "Genero", "Personal de Salud", "Lote"] + _opc_hdrs + ["Realizados", "Faltan", "Avance %"]

    tr1  = '<th class="s-idx" rowspan="2" style="vertical-align:middle;position:sticky;top:0;z-index:22;background:linear-gradient(135deg,#0a1a5c,#1C398E);">#</th>'
    tr1 += '<th class="s-dni" rowspan="2" style="vertical-align:middle;position:sticky;top:0;z-index:22;background:linear-gradient(135deg,#0a1a5c,#1C398E);">DNI</th>'
    tr1 += '<th class="s-pac" rowspan="2" style="vertical-align:middle;position:sticky;top:0;z-index:22;background:linear-gradient(135deg,#0a1a5c,#1C398E);">Paciente</th>'
    for h in fixed_headers:
        tr1 += f'<th rowspan="2" style="vertical-align:middle;position:sticky;top:0;z-index:21;background:linear-gradient(135deg,#0a1a5c,#1C398E);">{h}</th>'
    for idx_g, (titulo, ncols, color, _) in enumerate(GRUPOS):
        is_last_g = (idx_g == len(GRUPOS) - 1)
        border_right = 'border-right:none;' if is_last_g else 'border-right:2px solid rgba(255,255,255,.55);'
        tr1 += (
            f'<th colspan="{ncols}" style="background:{color};color:#fff;'
            f'text-align:center;border-left:2px solid rgba(255,255,255,.4);'
            f'border-bottom:2px solid rgba(255,255,255,.55);{border_right}'
            f'position:sticky;top:0;z-index:21;">'
            f'{titulo}</th>'
        )

    # Fila 2: sub-columnas de cada grupo (C0009 / C2062)
    tr2 = ""
    for idx_g, (_, _, color, id_cols) in enumerate(GRUPOS):
        is_last_g = (idx_g == len(GRUPOS) - 1)
        for idx_col, id_col in enumerate(id_cols):
            label = next((lbl for _cod, idc, lbl in ITEMS_CONFIG if idc == id_col), id_col)
            is_last_in_group = (idx_col == len(id_cols) - 1)
            if is_last_in_group and not is_last_g:
                border_right = 'border-right:2px solid rgba(255,255,255,.55);'
            elif not is_last_in_group:
                border_right = 'border-right:2px solid rgba(255,255,255,.55);'
            else:
                border_right = 'border-right:none;'
            tr2 += (
                f'<th style="background:{color};color:#fff;'
                f'border-left:none;{border_right}font-size:.6rem;width:140px;min-width:140px;'
                f'position:sticky;top:36px;z-index:20;">'
                f'{label}</th>'
            )

    ths = f"<tr>{tr1}</tr><tr>{tr2}</tr>"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Red San Pablo — Familias Sensibilizadas</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;font-family:'Inter',sans-serif;}}
:root{{--blue:#1C398E;--bg:#f4f6fb;--white:#fff;--border:#e2eaf4;}}
body{{background:var(--bg);min-height:100vh;}}

/* TOPBAR */
.topbar{{background:linear-gradient(135deg,#0a1a5c,#1C398E);color:#fff;padding:0 2rem;height:64px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 2px 16px rgba(10,26,92,.2);}}
.topbar-left{{display:flex;align-items:center;gap:14px;}}
.topbar-title{{font-size:.9rem;font-weight:900;letter-spacing:.06em;color:#fff;text-shadow:0 1px 4px rgba(0,0,0,.3);}}
.topbar-sub{{font-size:.68rem;color:rgba(255,255,255,.75);font-weight:600;letter-spacing:.04em;margin-top:1px;}}
.fecha-badge{{display:inline-flex;align-items:center;gap:7px;background:#fff;border:1.5px solid #d1d5db;border-radius:50px;padding:5px 14px;font-size:.72rem;font-weight:700;color:#374151;white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.08);}}
.fecha-dot{{width:8px;height:8px;border-radius:50%;background:#22c55e;flex-shrink:0;box-shadow:0 0 0 2px rgba(34,197,94,.25);}}
.btn-home{{display:inline-flex;align-items:center;gap:7px;background:linear-gradient(180deg,#ffffff,#dce8f8);border:1.5px solid #b0c8e8;border-radius:50px;padding:7px 20px;font-size:.75rem;font-weight:700;color:#0b1e42;text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,.15),inset 0 1px 0 rgba(255,255,255,.9);transition:all .18s;}}
.btn-home:hover{{background:linear-gradient(180deg,#f0f6ff,#ccddf5);box-shadow:0 3px 10px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.9);transform:translateY(-1px);}}

/* CONTENT */
.content{{padding:1.5rem 2rem;}}
.hero{{margin-bottom:1.2rem;}}
.hero h1{{font-size:1.6rem;font-weight:900;color:#0b1e42;letter-spacing:-.02em;}}
.hero h1 span{{color:var(--blue);}}
.hero p{{font-size:.8rem;color:#64748b;margin-top:4px;}}

/* FILTERS */
.filters{{background:#fff;border:1px solid var(--border);border-radius:16px;padding:1.1rem 1.5rem;margin-bottom:1.2rem;display:flex;gap:1rem;flex-wrap:wrap;align-items:center;box-shadow:0 2px 16px rgba(10,26,92,.07);}}
.filter-divider{{width:1px;height:36px;background:var(--border);flex-shrink:0;}}
.filter-field{{display:flex;flex-direction:column;gap:3px;flex:1;min-width:150px;position:relative;}}
.filter-label{{font-size:.6rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:#94a3b8;padding-left:2px;}}
.filter-icon-wrap{{position:relative;display:flex;align-items:center;}}
.filter-icon{{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:#94a3b8;pointer-events:none;flex-shrink:0;}}
.filter-select{{border:1.5px solid var(--border);border-radius:10px;padding:8px 12px 8px 32px;font-size:.82rem;font-family:inherit;color:#0b1e42;background:#f8faff;outline:none;width:100%;cursor:pointer;appearance:none;-webkit-appearance:none;transition:border .15s,box-shadow .15s;}}
.filter-input{{border:1.5px solid var(--border);border-radius:10px;padding:8px 12px 8px 32px;font-size:.82rem;font-family:inherit;color:#0b1e42;background:#f8faff;outline:none;width:100%;transition:border .15s,box-shadow .15s;}}
.filter-select:focus,.filter-input:focus{{border-color:var(--blue);box-shadow:0 0 0 3px rgba(28,57,142,.1);background:#fff;}}
.filter-select:hover,.filter-input:hover{{border-color:#93aee8;}}
.filter-arrow{{position:absolute;right:10px;top:50%;transform:translateY(-50%);pointer-events:none;color:#94a3b8;}}
.filter-actions{{display:flex;align-items:center;gap:10px;flex-shrink:0;}}
.btn-filter{{display:inline-flex;align-items:center;gap:6px;background:linear-gradient(135deg,#1C398E,#2563eb);color:#fff;border:none;border-radius:10px;padding:9px 20px;font-size:.8rem;font-weight:700;cursor:pointer;font-family:inherit;transition:all .15s;white-space:nowrap;box-shadow:0 2px 8px rgba(28,57,142,.2);}}
.btn-filter:hover{{transform:translateY(-1px);box-shadow:0 4px 14px rgba(28,57,142,.3);}}
.btn-filter:active{{transform:translateY(0);}}
.btn-clear{{display:inline-flex;align-items:center;gap:6px;font-size:.78rem;color:#fff;text-decoration:none;padding:8px 16px;border-radius:10px;border:none;background:linear-gradient(135deg,#ef4444,#dc2626);transition:all .15s;white-space:nowrap;font-weight:700;box-shadow:0 2px 8px rgba(220,38,38,.3);}}
.btn-clear:hover{{background:linear-gradient(135deg,#dc2626,#b91c1c);box-shadow:0 3px 12px rgba(220,38,38,.4);transform:translateY(-1px);}}
.btn-excel{{display:inline-flex;align-items:center;background:linear-gradient(135deg,#16a34a,#15803d);color:#fff;border:none;border-radius:9px;padding:9px 18px;font-size:.82rem;font-weight:700;cursor:pointer;font-family:inherit;transition:all .15s;white-space:nowrap;}}
.btn-excel:hover{{transform:translateY(-1px);box-shadow:0 4px 12px rgba(22,163,74,.3);}}

/* MULTI-SELECT */
.ms-wrap{{position:relative;width:100%;}}
.ms-trigger{{display:flex;align-items:center;gap:6px;width:100%;border:1.5px solid var(--border);border-radius:10px;padding:8px 10px;font-size:.82rem;font-family:inherit;color:#0b1e42;background:#f8faff;outline:none;cursor:pointer;text-align:left;transition:border .15s,box-shadow .15s;}}
.ms-trigger:hover{{border-color:#93aee8;}}
.ms-trigger:focus,.ms-wrap.open .ms-trigger{{border-color:var(--blue);box-shadow:0 0 0 3px rgba(28,57,142,.1);background:#fff;}}
.filter-icon2{{color:#94a3b8;flex-shrink:0;}}
.filter-arrow2{{color:#94a3b8;flex-shrink:0;margin-left:auto;transition:transform .15s;}}
.ms-wrap.open .filter-arrow2{{transform:rotate(180deg);}}
.ms-label{{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;}}
.ms-dropdown{{display:none;position:absolute;top:calc(100% + 4px);left:0;min-width:100%;max-width:340px;background:#fff;border:1.5px solid var(--border);border-radius:12px;box-shadow:0 8px 24px rgba(10,26,92,.13);z-index:500;overflow:hidden;}}
.ms-wrap.open .ms-dropdown{{display:block;}}
.ms-search-wrap{{padding:8px 10px 4px;border-bottom:1px solid var(--border);}}
.ms-search{{width:100%;border:1.5px solid var(--border);border-radius:8px;padding:5px 9px;font-size:.78rem;font-family:inherit;outline:none;color:#0b1e42;}}
.ms-search:focus{{border-color:var(--blue);}}
.ms-list{{max-height:220px;overflow-y:auto;padding:4px 0;}}
.ms-list::-webkit-scrollbar{{width:4px;}}
.ms-list::-webkit-scrollbar-thumb{{background:#cbd5e1;border-radius:2px;}}
.ms-opt{{display:flex;align-items:center;gap:8px;padding:6px 12px;cursor:pointer;transition:background .1s;font-size:.8rem;color:#0b1e42;}}
.ms-opt:hover{{background:#f0f4ff;}}
.ms-opt input[type=checkbox]{{accent-color:var(--blue);width:14px;height:14px;flex-shrink:0;cursor:pointer;}}
.ms-opt span{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.ms-select-all-row{{border-bottom:1px solid var(--border);background:#f8faff;}}
.ms-all{{font-weight:600;color:var(--blue)!important;}}
.ms-all span b{{color:var(--blue);}}
.ms-all-btn{{background:none;border:none;width:100%;text-align:left;cursor:pointer;}}

/* METRICS */
.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:1.2rem;}}
.mc{{background:#fff;border:1px solid var(--border);border-radius:12px;padding:.8rem 1rem;}}
.ml{{font-size:.65rem;font-weight:700;color:#94a3b8;letter-spacing:.05em;text-transform:uppercase;margin-bottom:4px;}}
.mv{{font-size:1.4rem;font-weight:900;color:#0b1e42;}}

/* TABLE */
.table-wrap{{overflow:auto;border-radius:12px;border:1px solid var(--border);background:#fff;max-height:65vh;}}
table{{border-collapse:collapse;border-spacing:0;font-size:.72rem;width:100%;}}
tbody td{{white-space:nowrap;}}
thead tr:first-child{{background:linear-gradient(135deg,#0a1a5c,#1C398E);color:#fff;}}
thead tr:last-child{{background:linear-gradient(135deg,#0f2472,#1C398E);color:#fff;}}
thead th{{padding:5px 10px;font-weight:700;font-size:.63rem;letter-spacing:.04em;text-transform:uppercase;border:none;border-right:1px solid rgba(255,255,255,.25);white-space:normal;word-break:keep-all;min-width:80px;position:sticky;top:0;z-index:21;}}
thead th:last-child{{border-right:none;}}
thead th.s-idx,thead th.s-dni{{box-shadow:inset -1px 0 0 rgba(255,255,255,.4);border-right:none;}}
thead th.s-pac{{box-shadow:inset -2px 0 0 rgba(255,255,255,.4);border-right:none;}}
thead th[colspan]{{white-space:normal;text-align:center;min-width:280px;max-width:280px;border-bottom:2px solid rgba(255,255,255,.55);}}
thead tr:last-child th{{top:36px;z-index:20;border-top:none;}}
thead tr:first-child th[rowspan]{{border-bottom:none;}}
tbody tr:hover td{{background:#dbeafe!important;transition:background .12s;}}
tbody td{{padding:6px 10px;border-bottom:1px solid #c8d6e8;border-right:1px solid #c8d6e8;vertical-align:middle;}}
tbody td:last-child{{border-right:none;}}
tbody tr:last-child td{{border-bottom:none;}}
.idx{{width:36px;text-align:center;color:#94a3b8;font-weight:600;}}
.s-idx{{position:sticky;left:0;z-index:10;background:inherit;box-shadow:inset -1px 0 0 #c8d6e8;}}
.s-dni{{position:sticky;left:36px;z-index:10;background:inherit;min-width:90px;box-shadow:inset -1px 0 0 #c8d6e8;}}
.s-pac{{position:sticky;left:126px;z-index:10;background:inherit;min-width:220px;box-shadow:inset -2px 0 0 #c8d6e8;}}
.ok  {{color:#166534;background:#f0fdf4;text-align:center;}}
.no  {{color:#991b1b;background:#fff1f2;text-align:center;font-weight:700;}}
.na  {{color:#94a3b8;background:#f8fafc;text-align:center;font-size:.65rem;}}
.warn{{color:#92400e;background:#fffbeb;text-align:center;}}
.avok{{color:#059669;font-weight:800;text-align:center;}}
.avmd{{color:#b45309;font-weight:800;text-align:center;}}
.avno{{color:#dc2626;font-weight:800;text-align:center;}}
.num {{color:#0369a1;font-weight:700;text-align:center;}}
.grey{{color:#64748b;text-align:center;}}

/* EYE BUTTON */
.eye-btn{{background:none;border:none;cursor:pointer;padding:0 4px;vertical-align:middle;color:#94a3b8;display:inline-flex;align-items:center;position:relative;}}
.eye-btn:hover{{color:var(--blue);}}
.tip-title{{font-size:.6rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#94a3b8;margin-bottom:6px;}}
.tip-item{{display:flex;align-items:center;gap:6px;font-size:.75rem;color:#991b1b;padding:3px 0;border-bottom:1px solid #fee2e2;}}
.tip-item:last-child{{border-bottom:none;}}
.tip-ok{{font-size:.75rem;color:#16a34a;font-weight:600;}}

/* ── LOADING OVERLAY ── */
#rsp-loading{{
  position:fixed;inset:0;
  background:#f0f4f8;
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:20px;
  z-index:99999;
  transition:opacity .4s ease;
}}
.rsp-ld-logo{{
  width:64px;height:64px;border-radius:16px;
  background:linear-gradient(135deg,#1C398E,#2563eb);
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 4px 20px rgba(28,57,142,.35);
}}
.rsp-ld-spinner{{
  width:44px;height:44px;
  border:4px solid #e2eaf4;
  border-top-color:#1C398E;
  border-radius:50%;
  animation:rsp-spin .8s linear infinite;
}}
@keyframes rsp-spin{{to{{transform:rotate(360deg);}}}}
.rsp-ld-brand{{ font-size:.82rem;font-weight:700;color:#64748b;text-align:center;line-height:1.5; }}
.rsp-ld-brand b{{ display:block;font-size:.95rem;color:#1C398E;font-weight:800; }}
.rsp-ld-brand small{{ font-size:.72rem;color:#94a3b8; }}
.rsp-ld-txt{{ font-size:.72rem;color:#94a3b8;letter-spacing:.06em;text-transform:uppercase; }}
.rsp-ld-bar{{ width:200px;height:4px;background:#e2eaf4;border-radius:4px;overflow:hidden;margin-top:4px; }}
.rsp-ld-fill{{ height:100%;background:linear-gradient(90deg,#1C398E,#2563eb);border-radius:4px;animation:rsp-fill 2s ease-in-out infinite; }}
@keyframes rsp-fill{{ 0%{{width:0%}} 50%{{width:75%}} 100%{{width:100%}} }}
.rsp-spinner{{ width:44px;height:44px;border:4px solid #e2eaf4;border-top-color:#1C398E;border-radius:50%;animation:rsp-spin .8s linear infinite; }}
@keyframes rsp-spin{{ to{{transform:rotate(360deg);}} }}
</style>
</head>
<body>

<!-- LOADING OVERLAY -->
<div id="rsp-loading">
  <div class="rsp-ld-logo">
    <img src="data:image/png;base64,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" style="height:48px;width:auto;object-fit:contain;mix-blend-mode:screen;" alt="Logo Red San Pablo">
  </div>
  <div class="rsp-ld-brand">
    <b>Red de Salud San Pablo</b>
    <small>Cargando datos...</small>
  </div>
  <div class="rsp-spinner"></div>
  <div class="rsp-ld-txt">Por favor espere</div>
  <div class="rsp-ld-bar"><div class="rsp-ld-fill"></div></div>
</div>

<!-- TOPBAR -->
<div class="topbar">
  <div class="topbar-left">
    <img src="data:image/png;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAHkAeQDASIAAhEBAxEB/8QAHQABAAMBAQEBAQEAAAAAAAAAAAYHCAUEAwIBCf/EAGAQAAEDAwEEBgUGCAcJDgUFAAECAwQABREGBxIhMRNBUWFxgQgUIpGhFTJCUrHBIzNicoKSotEWJDdDo7LCGFNjdZOz0uHwFyY1RFRWZGVzdJS0w9MlNKTi8TZVg4Tj/8QAGwEBAAIDAQEAAAAAAAAAAAAAAAUGAwQHAgH/xABCEQABAwIDBAcHAQcEAQQDAAABAAIDBBEFITEGEkFRE2FxkaGx0RQiMoHB4fAVIzNCQ1JT8RY0YnIkJUSSojVjgv/aAAwDAQACEQMRAD8AxlSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSleu2W2fc3+gt8N+S51htBOPHs86+taXGzRcry57WDecbBeSlWBZNll5lbq7nJYgIPNI/COe4cPjU2tGzjTMHdU9HdnOD6UheRn80YHvzUxT4DWTZlu6Ov01UBV7T4fT5B28f+OfjoqOjRpEp0NRmHX3DyS2gqPuFSK26C1TOwU2xcdB+lIUG8eR4/Cr5hxIsJroYcZmO39RpASPcK+9TUOzEQ/evJ7MvVV2o20mdlDGB25+iqKDsknLwZ13jM9oZbLn27td6FsqsLQBky50hXXhSUJ9wGfjU/oUkAEgjIyMjnUnHglDH/BftuVDTbR4lLrJbsAH3UXi6A0nHwRakuKHW46tXwJxXSZ01p1nHR2O2gjrMZBPvIqdWfQWsbvHak2/T051h1IU26UbiFpPIgqwCO+ubqWw3XTlzNtvMX1WWEJWW99K/ZPLikkVsRR0QduRht+Qtdas02IlnSSufu8zey4bVvgNDDUGM2PyWkj7q+6Wmk/NaQPBIr6ISpawhCSpSjgADJJqz9L7ENWXaMiVcHI1oaWMhD+VO4/MHLwJB7qyT1FPStvIQ0LDTUtVWu3YWlx/OKq0pSrgUg+Ir5qjRlfOjtK8UA1ecn0e7klsmNqWI4vHAORlIGfEE/ZVYa00hftITkRL1E6LpAS06hW826Bz3VfccHlwrFT4hS1Lt2N4J/Oaz1WFV1E3flYQOeo8FEXrJZnvx1pgOfnxkH7RXhk6O0vIGHLJDT/2aNz+riu9U1Rsq127b48+PYzIjyGkutqbkNk7qgCPZKs5weyvc7aVlumDRfnb6rHTOrpL+zlxtyv9FS8rZnpV7PRsSY2f72+Tj9bNcSdskiqyYN5eb7EvNBfxBH2VbN3ts+0XF233OK5FltY6RpwYUnIBHwIPnXkrA/CaGYX6MfLLyWzHjmJQOt0puOefndUlcdl2o44Koq4kwdQQ5uq9ygB8ajNz09fLbkzbVLZSOay2Sj9YcPjWk6VHTbNUzv3bi3x/O9S1PtjVsylaHeB9PBZYpWjrtpbT91yZtqjLWebiE7i/1k4NQ29bJ4bgK7RcXWFdTb430+8YI+NQ9Rs5VR5xkOHcfH1VgpdrqKXKUFh7x3j0VR0qR3zRGpLQFLet6n2U/wA7H/CJ8eHEeYFRyoSWCSF27I0g9askFTDUN3onBw6jdKUpWJZkpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlK7ul9KXnULg9Rj7rAOFSHfZbHn1nuGayRRPlcGMFysU08cDC+VwAHErhV39OaQv19KVw4akRz/Pvew35HmfIGrU0ts6sto3H5iflGUOO86n8Gk9yOXvzUzAAAAGAOQqz0WzTne9UG3UPVUzEdsGtuykbfrOnyHrbsUB09svs8IJdurq7g8OJT8xseQ4nzPlU4hxIsJhMeHHajtJ5IbQEgeQro2u23C6y0xLbCkTH1cm2GytXuHVVm6Z2FapuKEu3aTFs7ah81X4V39VJx+1mp4ew4a22TfP1Kq5/U8Xff3n+Q+gVUV3dJaQ1FqpbqbFbjLDJAdV0iEJRnlkqI7D7qsPWWwq6Wi0u3Cz3VN1LKStyOWOicKRzKfaIUe7h3ZNQjZfq6Ro3VTFyRvLiOfgpjQ+m2Txx3jmPDHWa9CtFTA59IQ4jnf7FeDhzqSpZHXtLWnlbTnfMZcV1tU7JdU6c0w7fZ5hrbZUnpWWHFLWhJ4bx4AYBxyJ51AK3IDAvVnyC3LgTWPFLra0/YQax9tJ0u/pDVsu0ObymAekiuH+caPzT4jiD3g1oYNir6sujm+IeX2UptDgbKBrJqfNhyPHP7rQ2xrSOnW9B2a5O2SA7PfYDq5DrCVrJJJHE5xwxyqmvSKlCRtRmMJxuxGGWQByHsBeP260fs+Y9W0HYGMYKLbHB8ejTn41lDalM9e2i3+RnI9fdQk9oQrdHwTWhgrnTV8sjje1/EqU2iaynwuGJoAvbwatLbDJXrmyuxuE5KGltHu3HFJHwAqovSni9Fre3ywMB+3pSe9SVr+4irA9GKV0+zlxgnjGnut47ilCv7RqOellFyzp+cB81T7Sj47hH2KrBRfscZc3mXfUrZxL9vs+x/IN8LBcn0YdMR7jeZuopjSXE2/dbjJUMgOqySrxSBw/Oz1Vcu0rWETRWm1XWQyZDq1hqOwFY6RwgkZPUAASTUF9FRxs6NujI/GJuBUrwLaAPsNeb0rm3Dp+yOj8WmWtKvEo4fYa81Tfa8X6KXS9vkBfxXqieaDAemh+K179ZNr/L6Lw6Q28ypeoGYmobbCjQH3AgPsKUCxk4ClbxII7eWBx7q7O1/XWzy8aUnWR26CdJUnejqislzo3R81QVwTjPA4PIms3toU4tKEJKlKICQOZNdm/aS1NYt43axzoqE83FNEt/rjKfjU2/BqNs7XtO6RoAdbdqrke0OIPpnxvG+DqSL2B4ZW+V1xm0KccShAypRAA7TW6IEdMSDHio+ay0lseAGPurF2honr2tLJDIyHp7CFeBcGfhW2KjNqH+9GztPkpnYqP3Zn9g81jvbFL9c2n397Od2WWv1AEf2a7mybZZN1k38pzn1wLQlRSHEpy4+RzCM8AB9Y548MHjiHzUu6g1q+hlQLtyuKggn6zjnD7a2bZrfFtNpi2yE2G48VpLTae4DHvrcxWvfQU0cUeTiO4AKPwTDI8UrJZ5s2Ak25kkqDxtjGz5pgNuWl99WPxjkx0K/ZUB8Kjup9gllksrc09cZMGRj2W5B6Vo92cbw8ePhUK2sbVNQytVy4Niuj1vt0J1TKPV1bqnVJJBWVcyCc4HLGKnfo9a+u+plTbLfHvWpMZoPMyCkBSkZCSlWBg4JHHnxOaj3xYpTQe1GXrIvfwOSlmT4LWVPsQhA1AIAFyOsZ9ioHVFguumrw7arvGLElviOOUrSeSknrB/241zK0z6TVjjzdDJvO4BJtr6ML6y2shJT7yk+VZpYadffbYYbW664oIQhAypSicAAdZqxYZXe2U4lOR0PaqjjOG/p9WYWm4OY7CvxXCv+krBe95U2A2Hj/PNew5ntyOfnmr6sewXUkyCmRcblCtzi05DBBcUnuVjgD4E1yNU7GtZWRlciOwzdo6BkmGolYH5hAJ8s15dX0FQeic8Ht08clkZheKUo6djHN6xr3DNZW1JstuMXeesshM5oceicwh0efJXw8KgMyLJhSFR5cd2O8n5yHElJHka1EtCkLUhaSlSThSSMEHsrnXuzWy8xvV7lDakI+iVDCk+ChxHlUbWbNxSe9Ad08uHqPFTGH7XzxWbVN3hzGR9D4LNFKsjVWy6XGC5NheMtocfV3CA4PA8lfDzqu5DL0d9bEhpbTqDhSFpKVJPeDVTqqKekduytt5d6vFFiNNXM3oHX6uI7QvnSlK1VvJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlem2wJlyloiQIzkh9fJCBk+PcO+pDorRNz1GtL5BiwAfafWn53cgdfjy+yrp07YbZYIYjW6OEZ+e4eK3D2qPX9lTeG4JLV2e/3Wc+J7PVVzF9o4KC8cfvP5cB2+nkoZo/ZlFihEu/lMp/mI6T+DT+cfpH4eNWI022y0lpptLbaRhKUjAA7AK+8WO/LktxorLj77iglDbaSpSj2ADiTV2bONhzz/RXHWK1Mt8FJgNK9tX/AGih83wHHvFWsmiwmLl5n87lRgMRx2bi7waPTzPWqPqc7ELPp2/a3btWomXHm3WVKjoDpQlTieO6rHHG6FciOVTj0kbDpS3W62rtrkKBco4DIhMpALjJyckDlg54nGcniTiqasVykWe8w7rEOH4jyXkd5Sc4PceVZI5zX0hdHdpINu1YpaYYXXtZNZ4aQTxBHHL1W1LdbLZYrctm02xmMy2kq6GM0lJWQPLJPfVKXP0gX/lJCLbpxHqiV4X6w8elWM9QTwSf1qvCz3CPdbTEucRW8xKZS82e5QyPPjUatWz7Rtnvrt5RbmVT5EhTqFyF7wQtRzhtJ4DBPDAyO2qRRy00bnmqYXO4dvG66RXwVcrYxRSBjeOQ04Wy9FLWV9IyhzdUneSFbqhgjPUe+sVa3TERrO9ogbvqonvhnd5bnSKxjuxWx9RwJVzs8iBDuTttceSUGQ0gKWgHnu55Hv8Adx41lTaVs5veiXw5IxMtzisNTGkkJz9VY+ir3g9RPGpXZqSJkjwXWJ0Cg9sYppImFrLtbck8r+P0Vg+jVrjdUdG3N7grLlvWo8jzU19qh591TPb7o7+E2klTobW9c7YFOtYHFxv6aO/gMjvGOustxJD8SU1KjOqafZWHG3EnBSoHII861/sq1gxrPSjM/KUzmcNTGh9FwDmB9VXMeY6qy4xTPoqhtbDzz7fQ/mqw7P1jMRpXYdUa2y7PVvDq7FIbehMCyx23PZRHjpSruCUj91YhnSFy5r8pz57zinFeJOfvrY+0i6xrTom8vOyWWXTBeDKVrCStZQQkAHnxxWMqybMMO7JIeJH1WHbOQb0MQ4A/T0V/+idK3rdf4JP4t5l0D84KB/qCuz6UUXptn8WSB7Ue4IJP5KkLB+OKqDZBrtrQtwuEl+C7NRKZSgIQ4EYUFZBJIPUTXa2j7YHNX6bkWMafbhtOqQrpTJLiklKgeA3QOrHnXqbDqj9UE7G+7cZ3HKxXiDFqT9ENLI/37EWseZI4WXy9HrWMfTWp3rfcnwzb7mlKCtXzW3QfYJPUDkgnvGeArQ2ttM27V2n3rNct8NLIWhxsjebWOShnzHgTWK6mmldqGs9OxUQ4dzD8RsYQzKbDiUjsBPtAd2cVsYng8k8wqKd1nenFa2DY/FTU5pKpu8w38dRbkri0dsPtFjvzF1nXZ65+rOBxlksBtO8DkFXE72DxxwqY7UNVxdJaSlT3XUiW4hTUNrPFx0jA4dg5nuHeKoiRt01w61uIFrYVj57cYk/tKI+FQDUN9vGoZ5nXm4PTZGMBTh4JHYkDgkdwArVZg1XVTNkrHggfnIBbsm0FBRU7osPjIJ4nzzJJ6lJdhkUzNqtkQoZCHFvE/mtqUPiBWq9Qy/ULBcZ2cerxXXc9m6gn7qx7oHVEjSGokXqLEYlOobU2EOkgYVwJ4ddWJqDbm9etL3KzvadTHcmRlsdM3LyE7wwfZKewnrr3jGG1FVVMext2gAajnmsWz+L0lDRPjkdZ5JIyPIWVc7PH2o2vbBIfx0SLiwVE8gOkHHy51tFe9undxvY4Z5ZrCI4HIrTOyPaxar1bI9q1BMbhXdpIb6R5W63JAHBQUeAUesHmeXPA+bR0UkobMwXtkfVetkcRhhL6eQ23rEenos2TmJEaa/HloUiQ04pDqVDBCgcEHzq9fRUsryG7vqB1CktuBMVgkcFYO8s/1R76s6/6D0dqKYLhdLHGkyFAEvIUpsr7CooI3vPNddItGnbMEj1S2W6KjhxDbbafsrTr8dbVU3QsaQ42v9lIYXsy6hq/aJHgtbe33UC9JS4tw9mjsNSh0k6S00lPWd1XSE/sD31S2wRMNW1Wz+ubuAXC0FDILnRq3fjy78V9dt2ukaz1E2iApfyVBBRG3hguKON5wjvwAAeodWTXZkbFdXQ4sO7WKbGlvbjb6UJWWXm14CuGfZ4Hr3hUjSRMo6DoJ3bjpL69Yt5WUTXTy1+Ke00zN9sW7pxAN/E3stC6ocuzWnpzthZaeuaWVGMh35ql9n7u+q72MbQ79qG/ztNamgpZnxWVPBYaLahuqSlSVpPI+0OPCvTs02osXWWdN6oSi26gjrLCt4gNvrScEA8kryPm8ieXYJBtB0LbtVMestrVbryyn+LXBglLiD1BRGCpP2dWKrrY2U29T1TLX0drbrHMK2OlfWblXRyX3dWaX6jyd2/dV/6TekoBtDerIjKGpiHksyinh0qFAgKI61AgDPYe4Vn2pjrTUutm48vRupbi7JTGkDfS8ApYUnkQvG8QQc8Sc8Kh1XTC4JIKcMkdvciOXBc6xuphqqsyRMLb6g8+KVxtS6Zs+oGNy4RgXQMIfR7LiPA9fgciuzSt6SJkrSx4uCo6GaSF4fG4gjiFQmstC3XT5XIQkzIA49O2nigfljq8eVROtTEAggjIPMVXmuNm8afvzrEERZXNUfk254fVPw8OdVHEdnS28lNmOXor3hG1jX2irMj/AFcPny7dOxU5SvvOiSYMpyLMYcYfbOFIWMEV8KqxBBsVd2uDhcHJKUpXxfUpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSvrFYelSG48ZpbrziglCEDJUewV9AJNgvhIAuV80gqISkEk8AB11Z+gdnCnOjuWomylHzm4Z4E96+z8339ldzZ5oJiyhu43RKH7jzQnmhjw7Vd/V1dtTxCVLWlCElSlHAAGST2Vb8KwENtLUjPgPX071QMc2nLyYKM5cXenr3L8NoQ22lttCUISAEpSMADsArs6NsqdQ6mg2Vc9mAJbnRh50EgHqAA5k8gOHEjjXtToLWioImp0xdCyRkH1dW9jt3fnY8qjqS4y6FJKm3EKyCOBSR9hqy77ZWkROF+qxsqf0b4XtdMw21sbi4Ww9BaB09o2MBbY3SzFJw7MeAU6vtAP0R3DzzVd7WNrF6i3uTpPTFufjzW3OhXIW3vOqUeXRI488jCjnOeAHOpxsa1mjWOk23X1p+U4eGZiesnHsr8FAZ8QR1VIp0Sx26XJ1NLixWZDUfD0xTeVpaTknjzxjPL7hVAbKYKtxq2b7hwPPh8l1N0IqaFgoXiNhzuBw49h5qlNDbFrjd3jeNczJDZeO+YyXN59wnrcWc48OJ8KqnXGnpOltUTbJKUFqjr9hwfTQeKVeYI4dRzVobR9uEuZ0tu0glcSOcpVOcT+FWPyE/QHeePhXB0Psl1Pq0Kul1eXbYzwKw/KSVPPqPEEJPHB+sfLNWeknqYAZ61wa06N5fnzPcqZXUtJUltLhrC94ObufafrkOWqsj0Y9Q/KGkZFieXl+2O5bBPEtLJI9yt73iv16TWn1XDSMe+x0n1i1O5WRz6JZAPuUEHu41VGy+6SNAbUkR7x/FWwtUKcFHglJPBWewKCVZ7KszaHtm0ou1TbPbYb96EllbLi/xTOFAg8SN48+oedR09HNFiTZqdtwc8tM9c9M/qpelxCmmwd1PVvDXNu3PXLTLXLLuXG2P7YlM9DYtXyCprghi4LOSnsS6esfldXX2iz9c6y0PbLa9C1BcYclt9vC4aPwy3En8kZx3E48ax9SpKowCnlm6VpLeYH05KGpdqaqCnMLgH8ieXI8/zVdDUa7O5epK7C1Latyl5YRJILiR2EjP+3bX5tV4u1pDwtdymQenSEu+rvKb3wOQODxr62DT97v0joLNa5U5YOCWmyUp/OVyHmasexbBtUzN1d1mwLYg/OTvF5weSfZ/arfnrKWnbuTPHzzPdqoylw+tq39JTxnPiMh8jkFVD7z0h1Tr7q3XFc1rUVE+Zr8VpexbB9KQ91dzlz7msc0lYabPkn2v2qnNp0TpG1NpRB05bGynktUdK1/rKyr41EzbS0zMo2k+A/PkpyDY6rkzleG+J9PFY3hQZs1e5ChyJKvqtNFZ+AqRW/ZzrmcQGNMXFOeXTt9CP28VsRtCG0BDaUoSBgJSMAV+q0JNqJT8EYHbc+ilYti4B+8lJ7AB6rJzexvaGv51kbb/OmM/co162diGu3PnR4DX58ofdmtTUrWdtJWHSw+X3W23ZDDxqXH5/ZZhXsJ1slAUHbSs/VElWfimvG7sU18j5sCI7+bLR95FaqpXwbR1g5dy9HZLDzz71kp7ZBtDaSVfwf3wPqS2T8N/NcO46K1dbwVS9NXVtCea/VVqSP0gCK2jSszNpqkfE0Hv9VrybGUZHuPcO4/QLCLrbjSyh1tSFDmlQwRX5rdE2FDmt9HMiMSUfVdbCx7jUXvmzLQ13BMjT0VhZHBcUFgjv9jAPmDW7FtQw/vIyOw39FGzbFSgfspQe0W8rrKVr1Ff7WgN2293KGgckMSloT7gcV87terxdik3W6zp26cp9YkKcwe7eJxV6330frY7vLsl9lRjzDcpsOgns3k7pA8jUA1FsZ1vaUKdZiMXRpPEqhubysfmKAUfIGpODFcPldcOAPWLeP3UPVYHisDd1zSW9RuO7XwVdVr3ZBqyDqjR0ItSEqnxGEMzGSfbStIxvY7FYyD345g1keXGkRJC48uO7HeQcLbdQUqSe8HiK+tpuVwtM5udbJj0SS3811pZSod3h3VkxPD24hEADYjMFYsGxZ+FTlxbcHIjj/kK6dpGxa9z9VS7tp6RFcjzn1PrbecKFMrUcq44IKckntHLHDNXdp2JLgWCBCnyjLlR4zbbz5z+EWlIBVx48T21nvT+3rUkJlLV2t0O6BIx0gJZcPiQCn3JFfzVW3e/3OC5EtFuZtHSApU8HS66B+ScAA9+M9lQFTh2J1IZDIBZvHL/PgrRSYvg1G59RCXBztW2Phw8VwNv06JP2o3NURSVpZDbDi08lLSgBXuPs/o1Aq/q1KWsrWoqUo5JJySa/lWynhEMTYxwACotXUGpnfMRbeJPelXD6OmhPlW5DVV0ZzBhrxEQocHXh9LwT9vgagOzrSkzWOp2LTG3kM/jJLwHBpoHifHqHeRWwrTb4dotce3QWUsRYzYbbQOQSPv6yag8fxLoI+gjPvO16h91Zdl8H9pl9plHuN06z6D84qttp+x616i6W5WLorZdTlSkgYZfP5QHzT3jzB51nPUFlulgubltu8J2JJRzQscx2g8iO8cKvt3bjb2dfO25UdDmn0qDKZiM74WDxc7CjPVzwM91WBtA0hatZ2FcGahAeCSqLKAyplfUQesHhkdY8jWhS4jVYduR1Yuw6cx/jlqpOuwmixcSS0LgHtOY4H/PAjIrC+rtL23UkPo5aOjkIH4KQge2j947vsqjdVacuWnZ3q85rKFE9E8n5jg7j293OtLXGI/AuEiDKRuPxnVMup7FJJBHvFcy8WyFd4DkG4MJeYXzB5g9oPUe+pjEsIirW77Mn8+fb6qBwfH5sNd0cmbOI4js9FmSlSnXejZumpHSp3pFvWrDb+OKfyVdh+B+FRaqHPBJA8xyCxC6fTVMVVGJYjdpSlKVhWdKUpREpSlESlKURKUpREpSlESlKURKUpREpSvrFYelSW40dpTrziglCEjJUT1V9AJNgvhIAuV+oMSTOltRIjK3n3VbqEJHEmrx2faMjacjiTJ3H7m4n23OYbH1U/eeuv7s70czpyH6zJCXbm8n8IvmGx9RP3nrqXVeMGwYU4E0w9/gOX3XNdodoTVk09Ofc4n+r7ea+kSO/LlNRYzS3X3lhDaEjJUonAA781p3Zzs/sGz6y/L2oXYqrk2jfelPEdHGz9FGevqzzJ5c8VS+wURDtVs3re7jec6Pe5dJ0at348u/FaT2iaSi600+LRLlyIqEvpeC2cZJSCMEHmME+eKw49WESspi4tYcyRyWxsvh7XQSVbWh0gNmg6XsDfxXFum1vRcexP3GBc03F5CujaiNpUh11Z5AJUAd38rGPPAqg9S2HXuqrpK1JJ0rPSZKt8hmEpAx1YTjJ4dfEmtH6K2faX0khK7ZAS5LAwZcjC3j4Hkn9ECpIxLiPvOMsSmHXWuDiEOAqR4gcqiKfEYaFzjSsJ63cvkp6qwmoxJjRWyBtv4W6X53OqyDsz1VK0TrBmctLgjk9BOYIwS2Tx4fWSeI7xjrNa+Ydi3CAh5pTciLJaCkqHFLiFDh4gg1SHpR6YhtxIWqozSGpK3hFlFIA6QFJKVHtI3SM9hHZXP2I7U7fYNOSrNqSQ6lmIkuwlJQVqUCeLQ78nIzw4niABW/iEH6nTtrIG+9oR+cvJReFVP6NVvoKl3uag8Ptfz7VY+ktk2k9PXd66JjqmvqeUuOJGFIjpzkBKesj6xyeHVXg2j7YLHpvpINp3LtdE5BShX4Fo/lKHM/kjzIqpdo+12+6o6SDbyu1WpWQWm1fhXR+WodX5I4duaretmlwSWdwlrnXPL1+y1K3aOGmaYMNYGj+q3kPqe5dPVF9uWpL0/d7q6lyU9jeKUBIAAwAAOoDh21zEgqUEpBJJwAOurK2e7H9QalS3NuO9Z7aoZDjqMuuD8lHDh3nHdmr60Xs80tpNKV263pdlgcZcjDjvkeSf0QK2qvG6WjHRx+8RwGg+fotGh2bra89LMd0HO51Py9bKgdE7HtV6h3JExoWaErj0spJ6RQ/Jb5+/A76uHS2xjRtn3HZjDt4kp470o/g89yBwx3K3qsmlVerxqrqTbe3RyGX3V1odnaGkAO5vO5nPw0XyiRo8SOiPFYajsoGENtICUpHcBwFfWlKiVO6JSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIuZfrBZb9H6C82uLORjA6VsFSfzVc0+RqrdVbBbLLC3tO3B+3OnJSy/+Fa7gD84eJKquWlbVPW1FMf2TyPLu0WlV4bS1gtMwHr49+qxnrHQ2ptKOn5XtriY+cJlNe2yr9IcvA4PdUbrdrrbbrSmnUJcbWClSVDIIPURVZa22LaYvnSSbSDZZiuP4FOWVHvb6v0SPA1Z6PaVp92obbrHoqZiGxz23dSOv1HX5H1t2rL9fuMy9JkNx47a3XnVhDaEjJUonAAHbmpBrfROodIS+iu8IhhSsNSmvaZc8FdR7jg91R+O89HfbkR3VsvNqCkOIUUqSRyII5GrNHKyVm/GQQVTZYHwSdHM0gjULXOyHRTOi9MIYcShVzlYcmuDj7XUgHsTnHjk9dRP0itd/JFrOl7W9ifNb/jS0niyyfo9ylfZntFcbZhttI6K1ayORwS3cUp/ziR/WHmOZqzdXaL0rrqE1InModUpALE6KsBzdPLChkKHjkVRnsfSV3S1zSRe9xoeXyHJdLjkjrsMMGGOAIFrHUDj8zzWVtDacmaq1NEssMKBeXl1wDIabHzlnwHvOB11tFhpDDCGWxuttpCUjsAGBUd0JoewaMius2dhwuvfjZD6gp1YHIEgAAdwAFRLb5r5nT1jcsNufCrvObKVbp4x2jzUexRHAe/qGfVdVOxepbFCMhp9SV4wyiZgNG+aoPvHX6Adf5wWe9dzmblrW9T4xBYkTnnGyOtJWcHzHGvxZ9NXy72uddLdbX5MSAkKkOIHBPh2kDicchxNdXZnoi462vgiR95mEyQqXKI4Np7B2qPUPPkK1pp2y27T9nYtNqjpYisJwlI5qPWonrJ5k1O4jizMPa2KMXcLfIdfWfuqxhGBSYq508p3WG+fM9XUOPd2YamRo8yK5FlMoeZdTurQoZBFUftD0U/p18zIgW9bHFeyrmWifoq7uw/7Hd+2vZLvdPqPSkbjxXLgNp59q2x9qfd2VQMqOzKjuRpLSXWnElK0KGQoHqNfZYqbGqfebk4d4PI9S+QzVmztXuPF2nuI5jr/Csu0qXbRdHPacmesxgt22PK/BrPEtn6ivuPXURqj1FPJTyGOQWIXSqWqiq4hLEbgpSlKwrYSlKURKUpREpSlESlKURKUpREpSlEX9SCpQSkEknAA66uvZdo0WWMLpcWwbi8n2UkfiEnq/OPX2cu3PD2QaQ6RSNRXJr2EnMNtQ5n++H7vf2Vb1vhyrhOYgwmFvyX1htptAyVKPIVcMCwoMaKqYdnV1+neqBtNjZkcaKnOWjiOJ5D69y+FKk+vNC3/RshpF2jpUw6B0clklTSjjJTnqI7D2cOFRirRFKyVoew3BVLmgkgeY5G2I4FfSK+9FktSYzq2nmlhbbiDhSVA5BB7Qat+0bfb7GtyWLhZ4k6SlISJAcLe8e1SQCM+GKpyvTan48W5R5MqGiaw04lTkdaykOpB4pJHEZrBVUcFSP2rN6y2aLEKmjcegfu315d2auGdP2u7Q9Nv3a3IEK1726iLFX0K5A+kUknKgOR9oA9QPGursA2eajseoHdQXtpVvQGFMtx1LBW6VEZKgDwSMdfHOKlWnNr+gZNtaSuabSptsD1Z1hQCABjdSUgpIHVj3VE9ou3OMIzkDRqFrdWMGe83upQPyEHiT3qAx2GqvvV0odSxQBjT1WsO3Q9vcroW4ZA5lbNUmRzesG56hqOzIc15/Si1TFeTD0pFcS46y6JUvdOdw7pCE+OFEkeHbVFV+5Dzsh9x991brriita1qypSjxJJPM1aGyvZDctSdFdL50tutJwpCcYekD8kH5qfyj5A86noxBhVKGvdkPE9Sq8xqccrS6NuZ7gOsqGaJ0ffdX3H1Szxd5KMdNIc9lpoH6yvuGSeytF7O9kuntLdHMlpF1uicHp3kew2fyEch4nJ7MVN7HabbZLa1brVDaiRWhhLbY+JPMnvPE17qqWI43NVktb7rOXPt9FfMJ2cp6EB7/AHn8+A7B9dexKUpUKrElKUoiUpSiJSlcKdqWNGkrYQw46UEpKgQBmtOtxCmoWh9Q/dB/OC9NaXaLu0qMnViOqCo+Lv8Aqr8/ws/6B/Tf/bUUdqsJH83wd6L30L+SlFKjSNWNk+3CUkdzmfur13DUlvt+mJeoZYdREioUtwBIKzjqAzzJwK3aPGqGtduQSXOuhGXzAQQvJAAzK7VKppXpD6UB9mzXojvQ0P7dfNXpEaa+jYrsfEtj+1W97RFzUkMBxE/yj4K6aVSSvSKsH0dP3M+LiB99GfSJsSn0Jd0/cUNkgKWHEKKR246/fXz2mLmvv+n8R/tHw9VdtK/KFJWhK0HKVDIPaK/VZ1DpSlKIlKUoiUpSiL4zYsadFciTI7UiO6N1xp1AUlQ7CDwNUntE2GMvdJcNHOhlziowHl+yfzFnl4K4d4q8qVtUlbNSO3onW8itGuw6nrmbk7b9fEdhWFrjCl26a7CnRnY0llW6406kpUk94Nd7R2u9T6TO5Z7ktMcnKozo32ifzTyPeMGtP7Q9A2PWkLcntdBNQnDExpI6RHcfrJ7j5Y51mDX2iL5oy4er3RjejrJ6CU2CWnR3HqPcePlxq60OKU+Is6KUDe5HQ9n5dc6xLBavCJOmhcS3+oZEdv5Y+ClN02463mxSwybbAJGC7Gjnf/bUoD3VGdFaZvev9UKYbddcUtXSzZrxKujSTxUonmo9Q6/DJEXrTXo6X3TEnSyLLbGUwrowN+W0tWVvq63QfpDu+jy5YJ9VxbhtO59NHmeIGnWV4w0PxirbHWSkgZ2J16hw+qsDSWnrZpixsWi1M9Gw0MqUfnOK61qPWT/q5CuFtjb1U5oqSnSasSf58Nkh4tY4hvH0vjjOOOK+O2246mtuiJD2mo5Us5Ep9s/hGGscVJH2n6I494gOxTa3vdBpzVcn2uCIk9xXPsQ4fsV7+2qtTUlRIz21tn2OYOZ+au9ZX0sUn6c67A5tgRkByAP4OC6mxbaum79Dp3U7wbuYwiPKXwEjsSrsX/W8efh2+bNIJhS9X2dTEN1odJNYUoIQ7x+enqC89X0urjz/AJ6RGhLOzAd1hCkMW+XvgPsk4TKUTzSB9PrPaAScYJNSak1vqTUVmg2m63Bb8aGPZHIuHqUs/SIHAE/aSTNUVKJpW1dGd0H4h525/TwVcxKtNPA+gxBu+4C7HDwJ5EeOnWYpcYca4QnYUxpLrDyd1aFdYqgddaYk6ZupZVvORHSVR3sfOHYfyh/rrQ1czUtmiX60O26Yn2VjKFgcW1dSh/t3VJ4thja2PL4xofoobAsZfhs1nZsOo+o6/NZrpXvv9ql2W6vW6ajddaPAjktPUodxrwVzx7HMcWuFiF1iORsjQ9huDolKUryvaUpSiJSlKIlKUoiUpSiJUp2b6YXqO8gvpIgRiFPq+t2IHj9me6o/bIUi5XBiDEbLj76whA7+093XWitL2WNYLKxbowB3BlxeOLizzUf9uWKm8Ew32uXfePcbr1nl6qubR4v7BB0cZ992nUOfp9l0m0IbbS22lKEJASlKRgADkBWltgOzv5Bgp1JeWMXWSj8A0scYzZ+xahz7Bw6zUP8AR82d/KspvVd5YzAYX/E2ljg+4D88/kpPvPgc2ztjOqDoeW3pVnpJK/ZfKFfhUtYO90Y61dXbjOONS2MYh0sgoonWvkTw7PXuUDs/hXQRHEZ2k2F2jiev07+SqnbhrWTq2+NaI0yFSY6Xwh0tcfWXgeCQfqpPXyJ48gDUD1/oG/6Lea+U2UuxXQNyUzlTZVjiknqI48+fVV27Bdng03bhqK8shN2kt/g0LHGK0eruURz7Bw7arrblr93Vt4Tp+yLW5ao7u6Oj4mW7nGRjmkckjr59mM1BUbs4paUXjb8R5nn+eSwYpSb9Ka2ucRK/4WjgOVvPl2lVZSroc2E3A6JZmNTAnUGC67FWQGikjg2FdSh2nhk44DjVMOJKFqQrG8kkHBzU1TVsNVvdE69tVXKzDqii3enbbeFx+cxxC/lf1CVLWEISVKUcAAZJNfphl2Q+2ww2t11xQQhCBlSlHgAB1mtL7G9lUbTbbN7vraJF5UApts8URPDtX2nq6u04cRxGKhj3nZk6Dn9lsYThE2JS7rMmjU8vv1Lj7GtkKYYZ1BqyOFyeC40BYyGuxTg61fk9XXx4C7qUrn1XVy1chkkOfl2Lq1BQQ0MQihFh4nrKUpStVbqUpSiJSlKIlKUoiVW8/hOkD/Cq+01ZFVxcuFxkj/Cr+01Q9ux+xh7T5BbNPqV56UpXNltJX616N7YhqAdjR/rJr81I5Wnl3vZ5NsTrvq6p7C0pWU725n5pI6+QNW3Y2Nz65xA/hP0XqORsUsb3aBwPcViyldTVViuGmr/Ksl0bSiVGVuq3TlKgRkKB6wQQa5dXkgg2K60x7XtDmm4KUpSvi9retpO9aoiu1hB/ZFeqvFYTvWKArtjNn9kV7anxouIPycUpSlfV4SlKURKUpREpSlESvHeLZAvFtet1zityoryd1xtYyD+49hHEV7KV9BINwvjmhwsdFlLa3syn6OkqnwukmWRxXsPYypgnklzHwVyPceFQS2T5lsuDM+3yXI0phQW262cFJrccqOxKjORpLLbzDqShxtxIUlSTzBB5is1badli9MqXfLEhbtmWr8K1xUqKT39aOw9XI9pueE40JwIKjXQHn1Hr81zzHtnHUxNVSfCMyOI6x1eXZpamyDaZD1jETbrgW417aR7bXJMgDmtH3p6vDlCdtuyYMpkal0tHAbALkuCgfN6ytsdnanq6uyqQhyZEOU1KiPOMPtKC23G1YUlQ5EGrB1ftd1DqLSLFicSiMtSSmdIaODJHUMfRB+kBz7hwrL+lTUtUJaQ2adQeH5w5LB+uU9bQuhrxd7R7pGpP0PPgR1qF3e/Xi7xIUS5XF+UxBb6OMhxWQ2n7+zJ44AHICubSlWFrWtFmiyqj3uebuNylKVOLTsn15cremcxZC22tO82l95Da1D81RBHniscs8UIvI4DtNllgpZqgkQsLiOQuqa2l6WTqG0dLGQPlGMCpk9ax1oPj1d/iaoZSVJUUqSUqBwQRgg1rO92m5WS4LgXaE9Dko4lt1ODjtHaO8cKpPbLpgRJQ1BCbwy+rdlJSOCXDyV4Hr7/Gqzj+HiRntcXz6xz/ADgrlstiroX+wz5f034Hl8+HX2qt6UpVPV/SlKURKUpREpSlESlK7uhrCvUOoWIOCGE/hJCh9Fsc/M8B51kiidK8MYMysU8zII3SvNgBcqw9jGmxFgm/y2/w8gFMcEfNb61eJ+zxqx6/DLbbLSGmkBDaEhKUgYAA5AVZ+idjOpdQwG7jKdYtMR1IU106Sp1aTyUEDkPEiujRCnwynax7rAeJ4rkcxqsZq3PjaXE8OQ4Ll7MNpF30VKDIKplpWrLsRavm9qkH6KvgevtGodJ6jtGqLSi52aUl9lXBaTwW2r6qh1H/AGGRVCX3YJqOIwp21XOFclJGeiUCytXhnKfeRUEs901Rs+1KVtJkW6c1gPR3kEJcT2KT9JJ6j5g1FVlFSYoDJTOG/wCfaNfmpugxGvwQiKsYej8uw6fJan2m2a83/Rs212OeiHKeTglQ4Oo62976Oe3j2ddVzsI2XyLTMVqLU0QtTWlqRDjLwejIOC4e/wCr7+sVN9mW0Wz61hhtsiJdG05ehrVx71IP0k/EdfVnzbZtfNaMsXQxFoXeJaSIzZ49GORcUOwdXafA1BwOrYg6ga2xcfn38vorLUsw6YtxR7rtYMuXVlz6uaiXpC7RPUWHNI2V/wDjTqcT3kH8Ugj8WD9Yjn2Dh18M9JBUoJSCSTgAczX7kPPSZDkiQ6t151RW4tZypSickk9Zq9/R32dpDbWsL3HBUfatzLg5D+/Ef1ff2GrT+wwak5nzP53BUo+07Q1/IeDW/neV2thezEWBhvUV+YBuzqcsMLH/AMqkjmfyyPcOHPNW5SlUWqqpKqQySHMrplFRRUUIhiFgPHrKUpStdbSUpSiJSlKIlKVxrhqCLDluRlsvLUjGSnGOWe2tWrraejYHzu3QTb5r61pdkF2aVHv4VRP+TP8Aw/fX8Oq43VFe94qNO0uFj+cPH0Xvon8lIqrm6cLnK/7Zf9Y1IzqtjqiO/rCozMdD8x58JKQ44pYB6snNVDa3FaOuijbTv3iCb68usLPAxzSbr40r9ISpa0oQkqUo4AHMmum/YLm00HOgCxjJShWSPL91U6CjqKhrnRMLgNbC9lnLgNV4rc2l64R2l/NW6lJ8CasYcBgVwtMWcRWxKlIHTq+akj5g/fXerqOyWFy0VKXyizn2NuIHC605nhxyWV/SmUyraagNfPTbmg7+dvLP2FNVRU426x7jH2p3r5TkNPuuOhxtTasgNEDo0kdRCcDB7M9eag9b8xvIV1zCmBlFE0G/ujySlKVjUgt3aYO9pu1q7YbR/YFdGsw27b9qGDbY0FqyWtSY7KGkqUXMkJAGT7XdX1V6Q+qvo2WzDxS6f7dSwq47armL9mMQc8kNHeFpmlZiV6Q2r/o2mxDxadP/AKlfxHpC6y3wV2uwlOeIDLoJHj0lPbIl8/0riHId609SvLapYn2uJOSjcElhDoTnON5IOPjXqraVdILTYpSlKL4lKUoiUpSiJXzfaakMOMPtodacSULQtOUqSRggg8xX0pRFlvbXs1d0nNVdrU2tyyPr4DmYyj9BX5PYfI8eJrOt03CHFuEF6FNYRIjPoKHW1jKVJPMGsl7XdDP6K1EWmwty1yiVw3jx4daFH6yc+Ywau+B4v7QOglPvDQ8/v5rm20mAilJqYB7h1HI+h8FCqk2gtEXzWciS3aWkBuM0VuOunCN7B3UZ7VEeXOozWpvRujRGdmEZ6OB00iQ8uQevfCikfspTUji9a6jp+kYMybKKwHDmYhV9FIfdAJP581S2xOzNyNrNvt92jlKoy3VrYcTycbSogEHsUM+Vad1Zf7fpixP3m6KcEZjdCujRvKJUQAAPE1Rm0TUdqtHpAQblBjGOuE621cXs4Du8MLOO5CsZ68d2Td2u7KnUej7pZiAVSY6g1nkHB7SD5KAqtYu/p5oJZQQ1wHyzz8CrhgMfs1PUwQEF7HGx55ZeIKjGsLJYtq2g0TrW42t/dUuDJKcKQsc21dYBIwR4HqFZMvNuRIYlWu4sHdUFMvNLGCDyI7iD7iK1T6NzMKPs5CI0svvqluLltkYLDnBO5jwSk+dUhtzajM7Vb4mKUlBdQtW6OSy2kr894mpLB39HUS0WrBe1+23jf8uojaCPpaWDEcg91r27Lg/K35ZY31TZn7DfJFtfyejVltePnoPzVe745rl1d21/TvyrY/lOMjMuCkqOBxW19IeXP39tUjVcxWhNHUFg+E5js+yt2CYkMQpRIfiGR7fvqlKUqNUulKUoiUpSiJV57I7D8kacTMfRiVOw6rI4pR9Ae458+6qr0BZTfdTxYa070dB6V/8AMTzHmcDzrRsCK5MmR4UcJ6R5xLTYUoJGScDJPACrVs3Rgl1S/hkPqVSNr8QIa2jZqcz9B359ysn0ftEp1LqQ3W4M79stqgopUPZde5pR3gcz5Drq+td6709oxtn5YfdLz/FqOwjfcUkHBVjIAHiRnqzXr0JpyJpTS8OyxcK6JOXXAPxrh4qV5nl2DA6qrDbRsovF+u8jUllnKmvuJG/CfISUhIwA2rljuOOviSa1ZKmDEa79u7dYMh+cL6+C3YqOpwjDbUrN6U5u/ONtAPmrD0br/S2rD0VquKRKwSYr46N3A7AfneRNfvaDoy06ysy4U9pKJCUn1aUlPtsq7j1p7R1+ODWP5DNwtFyLT7cmDNjr4pUC242ocu8Gtg7LrlcrvoCz3G7g+uvsZcURgrAUQlf6SQFedfcTw79OLZ4H5Xy5/cL5g2L/AKuH0tVHmBnyPD5H8yWRJLVy05qB1jpXIlwt8hSCtpRBQtJxkHyr+agvNzv91dul3lrlS3cBS1ADgBgAAcAO4V3dsUqPM2nX56MQWxKLeRyKkgJV8Qa4mmrNN1BfYlnt6N+RKcCE9iR1qPcBknuFXGN7TE2eQWNrk8uJXPpY3iZ1NESRvWA5m9h81M9iOgV6vvvrk9pQs0JYL5PDpl8w0D8T2DxFaqbQhttLbaEoQkBKUpGAAOQArmaTsUHTen4lmt6MMx0BJVjBcV9JZ7ycmurXPsTxB1bNvfwjQfnNdVwbCmYdThg+I5k9foOCUpSo5S6UpSiJSlKIlKUoiVXl7UVXiWT/AH1Q9xxVh1Xd4/4Wl/8Abr/rGqNt0T7NEP8AkfJbFPqV5KUpXM1tpSlKIpLpC1hRTcXuSSQ0ntPLNSquBoqR0kByOTxaXkeB/wBea79dm2Zhgjw2Mwj4hc9uh8rLRmJLzdK/LhIQopGVAHAr9UqfWJYJur82Vc5Mi5LdXMcdUp9Tud8rzxznrzXmq+fSO2bSROd1jYYhdZcG9cWWxxQofzoHWCPndhGes4oaoOWMxusV2PDa6Ktp2yR/McjySlKVjW+lKUoiUpSiLdGizvaOsqu23sH+jTXXri6DO9oawK7bZGP9Emu1U83QLiVR+9d2lKUpXpYkpSlESlKURKUpRErh640zA1bpyRZrgnCXBvNOgZUy4PmrHh8QSOuu5SvTHuY4OabELxJG2RhY8XB1WIdT2Sfp2+SrPc2ujkR17pxyUOpSe0EcRVo+jVrJq13R7S9wdDceesORVKPBL+ACn9IAY7wB11PvSC0UNR6bN4gs710tqCoBI4us81I7yPnDzHXWX0qUhQUlRSoHIIOCDV+p5Y8Yoix+R0PUeB/Oxcuq4JcAxESR5t1HWOIP5yK0ttM2Ot6q1R8twLoiAqRuiWhbW8CQAN9OCOOAOB7M5q1YzQYjtMJKlBtAQCrmcDHGs7aK27XK2wG4WobeboGwEpkoc3HSPysghR7+HfnnX411tyuN3trtusNvNrbeSUOSFub7u6eYTgAJ4dfE9mKg5sKxKYsgkza3IHK1vNWWDG8Hpw+piye/MixvfyUZha3uGjtf6hm2BbS4sqVIQGl5Lak76ujVjtTkEd2R11C58uTPmvTZjy35D6y464s5KlE5JNfGv6hClq3UJKieoDJq4xwRxneAzsATzsufS1Uso3CfdBJA4C6/KgFApUAQeBB66z1tAsRsGpX4qEkRnPwsc/kHq8jkeVaQVbbimOqQq3y0soGVOFlQSB3nGKgG2CyfKemTNaRmRAJdGBxLf0x7sH9GovHKQVVMXN+JufqFN7N17qKsDH5Nfkfoe/zVG0pSufLqqUpSiJSleu0QXbndI1vY/GSHUtg9mTz8udfWtLiGjUry9wY0udoFbuxWzCFp9y6uow9OV7JPU2ngPecn3VP6+EGMzChMQ46d1phtLaB2ADAr711GjphTQNiHAePFcVxCrdWVL5zxPhw8FYOzzavqLSnRxH1m6WtOB6u+s7zY/IXzHgcjuFaI0PrrTusI29apgEkJy5Eewl5Hl1jvGRWNq+kSRIiSW5MV9xh9tW8hxtRSpJ7QRxFR9fgkFVdzfddzH1ClsL2kqqGzH++zkdR2H8C2ZrDR2ndWMJbvVuQ8tH4t5J3XUdwUOOO7lXN2p6pj6I0S4/GDbcpaRGgMgcArGAcdiRx8gOuqs2ebcpcTo4GrmlS2BhImtJHSp/PTyV4jB8ahW2XWR1jq1b8ZxRtkQFmGkgjKfpLweRUfgAOqoOlwapNQ2Kf4G59XYO1WSt2ioxSOnpcpXZaWI6z2cNc1CnFqcWpxaipaiSpROSSeutIejhor5Jsp1PcGcTbgjEZKhxbY558VcD4Adpqmdk+l1at1tDtq0ExGz08sjqaSRkeZwn9KthoSlCEoQkJSkYSkDAA7K29o67caKZnHM9nALR2RwwPcayQaZN7eJ+nev1SlKpy6AlKUoiUpSiJSlcW7agZgylRksKeWn5x3sAVqVldT0UfSVDt0afll6a0uNgu1SowdWJ6oB/yv+qv5/Cz/AKB/Tf8A21EnarCR/N/+rvRe+hfyUoqvb4MXiX/2qvtrsnViuqCP8r/qrgT5HrUx2QUbnSK3t3OcVVNqsZosQgYynfcg30I4dYCzQxuac18KUpVGWwlKUoi7ui3VIuim8EpcbIPcRxH31Mq8VkbQ3aYoQhKd5pKjgYySBxr212rZ+hdRULY3OvfPsvnZaErt510pSlTaxrh671BH0vpK43yRuqEZkltB/nHDwQnzUQKw/IdW++485jfcUVKwABknJ4DlV7+llqMrlWzSzDnstp9ckgHmo5S2D4DeP6QqhaiqyTefu8l0rZWh6Ck6Y6v8hp6pSlK1FaEpSlESlKURbh2dHe2facV22qKf6JNd6sm2fbdrK1WeFaorNq6CHHbjtFcdRUUoSEjJ3+eBXoVt712eQtSfCMf9KpRtZGAAubzbLVz5HOFrEnj9lqqlZRVt318eT1uHhFH76/P+7rr/AHgfWoGB1eqJwa++2xrx/pKv5t7/ALLWFK5mlLiu8aXtV2dQltybCZkLSnkkrQFEDu41062gbi6rT2lji06hKUpX1eUpSlESlKURKypt50V/BbVJmwmt21XElxkAcGl/Tb7hxyO446jWq6jW0vTLerNHTbQQnpynpIqj9B5PFPhnke4mpLCq40dQHH4Tkez7KHxzDBiFKWD4hmO3l89FkCySo8K8RJcuI3MjtPJU6wscHEA+0nzGa11aNH6Efgx5sHTNldYfbS60sw0L3kqGQeIPUax2804y8tl1CkONqKVpUMFJHAg1KLTtF1narXFtcC+PMw4vBpsNo4DOcFWMkceRPLhVxxXD5awNML7EdZsR8lz/AAPFIKBz21Ee8DpkCQfmtaR7BYo//wAvZbazj6kVCfsFfWRLtdsR/GJUOEj/AAjiWx8cVxIj8TX+zkOtOFlu6QyCpCjllzkePalY+FQDZJsfFtkJvmrm25E1K8sRCQtDZB4LWeSldYHIdfHlTWQRljzUSEOabWtcnxXQpKiVr420sQc1wvvXsAO7u5q27nDh3mzyIMgJeiTGFNq3TkKQoYyD55BrFeq7K/Y77cLHPQFLjOqaXkcFp6jjsIIPga1dtS17btEWjpHN2Rcn0n1WLnio/WV2JHx5Duyff7vcL7d5F1ukhT8uQreWs+4ADqAGAB3VP7NRTND3ke4fNVbbGenc5jGn9o3XqB59fJZh1haTZNRzLdg9G2vLRPWg8U/A/CuRVsbdLRvsQ720nig+rvEdhyUn37w8xVT1XsTpfZal0Y01HYVa8GrfbaNkp10PaNfVKUpWgpRKsDYha/WtQv3NacohNYSfy15A+G97xVf1e2yC3eo6MZeUnDkxan1duOSfgAfOpnAafpqxpOjc/TxVe2nq/Z8PcBq73e/XwUxpUv2Z6Au2uJ7iIq0xYLBHrEtxOQkn6KR9JXdw7yOFXNF2B6SRHCJFxvDruOK0utoGe4bhx5k1cKvF6WlfuPdnyCoNDgNbXM6SJvu8ybX7FmqlXrqX0f1pbU7p2+dIocmJqMZ/TT1/o1TOoLNc7BdHLbd4bkSU3zQvrHUQRwI7xwrNS4hT1f7p1zy4rXrsKq6H9+yw56jvC8FKV3tnthVqXWdsswBLb7wLxHU0n2ln9UHzrZlkbEwvdoBdacMTppGxs1JAHzWgvRz0r8h6O+V5Le7Nu2HeI4pZHzB55KvMdlWhX5bQhptLbaEoQgBKUpGAAOQFfquXVNQ6oldK7UrtVHSspIGws0aPw/NKUpWBbKUpSiJSlKIlQHUn/Dcr84fYKn1QLU3C+SfEf1RVL24H/hR/9voVnp/iK5tKUrly3EpSlESlKURKUpRFP9PSEyLPHUk8UICFDsI4V0Kien71BgxERltOpJJK3AMgn/8AFSeNIYktB2O6lxHak12rA8ShqqWNgeC8NFwNdOX4FoSMLSV9aUpU2saxJtOukq8bQL3OmIU26Zi2+jVzbSg7iUnwCQKjdW/6Tej02XUzeo4aSIt2UovJ6kPjif1hx8QqqgqDlaWvIK7Hhk8c9JG+LSw+VsrfJKUpWNb6UpSiJSlKIlKUoiUpSiLbmy872zfTR/6rjD+jTUjqMbKDvbM9OH/q1kfsCpPU8z4QuK1mVQ/tPmlKUr0tdKUpREpSlESlKURZj9JHSvyNq5N7jN7sO7ZWrA4JfHzx58FeJV2VVdbB2w6eGpNn9yhIbC5LKPWY3DJ6RHHA7yN5P6VY+roGAVntFNuu1bl8uHp8lyvaigFLWb7R7r8/nx9fmtb7FdKz9JaOTDuE5Mh2Q56x0aCChneA9lKuvlknlnl2n0bUde27RFo6RzdkXJ9J9Vi54qP1ldiR8eQ7q50Xtgg2bZU01NPrN6hExI8cn8akD2FqPUkDges7vfVQzXdR60v781TEy6z31ZUGWlLKR1AADgkdQqJgwiSoqny1WTQey/2U7U4/DS0UcFDm4gW47vb19X0Xj1DeLjf7u/dbrJVIlPqypR5AdQA6gOoV4KsnT+xXW1z3Vyo8a1NHjmU7lWPzU5Oe44qxNP7A7BF3XL1dJlxWOaGgGW/A81H3ipuXGKGmG6HXtwGf2VbgwDEqt2+WEX4uy88/BZe1ZbBeNOTrdgFTzR6PPUscU/ECs2qBSopUCCDgg9VbY2zaajaW15Kt0FkswXG0PRkFRVhChgjJyT7QUPKsk7Sbd8mazuDKU7rbq+nb7ML4nHgcjyqF2hY2eGKrZocu/MfVWPZSR9NUTUMmoz+YyPfko5SlKqivC+sNhyVLZjNDLjziW0jvJwK03AjNw4LENkYbYbS2jwSMD7KobZdB9e1vASRlDCi+ru3RkftYrUGzezfL+urRaijfbekpLo7W0+0v9lJq4bOMbFBJUO/ABdUDa6R09VDSs1+pNh5LUmy+ysaV2eW6I6EMrSx6zLWrhhahvKJPdy8Eiq+uHpBW5q5rZh6dfkw0qIS+qSEKWO0I3TjzPuq3NSWtF7sM20OSHo7ctlTK3GsbwSeBxkEcuHnVC6h2A3uPvOWS7xJyBxDb6SyvwHMHzIqMw40VRI99afeJy1A7wpnFhiVLFHHh7fdaM9CctBY/RWnoPahpjV0lMGK69DuChlMaSkJK8c90gkK8OfdXJ9I7T0W6aCeu3RpEy1qS425jiUKUEqT4cQf0aqXRGgNXWvaTYUXGyzIqG5zbqnwnebCUHfPtpyniEkc6uX0h7iiBsuntKIDk1xqO33neCj+yhVZpKaKlxCH2V1wSON+NjnysteKsnrcKn9tZYtBGluFxkeN1lKr29FbT+VXPUzyOWIcckeCnD/UGe81RNbL2X2ZNg0DZ7cEbjiYyXHgRx6RftKz4EkeQqY2jqeiphGNXHwGv0Vf2Ro+mrDKdGDxOQ+qktKUqiLpyUpSiJSlKIlKUoiVA9UcL7J8U/wBUVPKhupoE1y8vOtRXnELCSFIQVD5oHV4VUdtIXy0LAxpNnDTsKzwGzlwqV6/k24/8hk/5I1/Ra7if+IyP1DXMhRVJ/lu7itveHNeOle0Wm5H/AIk/+rX9+SLn/wAid91ehh9Wf5Tu4+ibzea8NK94s10P/EnPhX6Fkuh/4mv3j99fRhtYf5Lv/ifRfN9vNc6ldIWO6/8AI1frJ/fX7b0/dVqwY4QO1S0/vr23Ca9xsIXf/E+ib7ea5rTanXUNIGVLUEgd5qw7bDbgw0RmskJ4knmT1muZZLAiE6mRIcDryfmgfNT++u5XRdlcDkoGumqG2e7IdQ+/0WrNIHZBKUpVwWBeW52633ON6tcoMWaxvBXRSGkuJyORwoEZqivSM2axmYCdVact7EZuMjcnRo7QQnc6nQkcOHJXdg9Rq/6/D7TT7DjDzaXGnElC0KGQpJGCCOyscsQkbYqQw7EZaGZsjDkNRwI4rAdKubUWwPU3y7M+RHbebaXSqMXn1JWEHiEkbp4jlnrxmvEnYBrg85FmT4yV/wChUSaeTkumNxygc0HpRmqmpVup9H3Wx5z7EPGQ7/7dftPo9ay67pYB/wDzvf8At09nk5L7+uYf/dCp+lXIn0eNW/SvFkHg46f7FfRPo76n+le7OPAuH+zT2eXkvP69h390eKpelXWn0dtQ/Sv9rHghw/dX0T6Ot7+lqK3jwZWa++zS8l5/1Bh390ePoqQpV5p9HO6/S1NCHhGV++voj0cpxPt6qjAd0NR/t09ml5L4dosNH83wPord2Qne2YadP/QGx8KldcvSlnb0/pu32Vp5TyIUdLIcUMFeBxOOrJ6q6lS7BZoBXLqp7ZJnvboST4pSlK9LAlKUoiUpSiJSlKIlY62taf8A4Na+uduQjcjqc6eNgcOjX7QA8OKf0a2LVHelZZUrgWjUDaPbbcVEdV2pUCpHuIX+tU5s/U9DVhh0dl6fnWq1tVRiooS8asz+Wh9fkq/2BxrHP2gNW6+W+PNbkMLDCXhlKXU+0DjkfZSoYPbWqoUSLCYTHhxmYzKfmttNhCR5DhWKNLXRdk1Jbru3nMSSh0gfSAIyPMZHnWxNXQlXzRtyhwnlByXDX6u42og7xTlBBHVnFbu0cR9oYS6zXeFv8qO2Rnb7LI1rbuab9ZBGQv2gr+X/AFbpqwhXyve4UVaebanQXP1BlR91V3qHb3pyJvIs1um3Nwclrwy2fM5V+yKzcrO8d7O9njnnSpCDZumZnIS7wHr4qKqtsKyTKJoYO8+OXgpTtI1tP1xc2Js6FEimO2W2wyFZ3Sc4USePHPZzqgdvFvAXbrqkcwqO4f2k/wBqrVqJbWoPruiJagMrjKS+nyOD+yTW7iVIz2B8TBYAXHyzUdhFfJ+qRzSG5cbE9uSoWlKVzldcVlbBom/c7lOI/FMoaB/OOf7FXnpLUVz0vekXa0raRJQkoBcbCwUnmMH7RxqqthsUNaXkySPaflHj+SlIA+Oa0Lsz2WSdb6ekXZm7twizJMdLa2CsLwlKs7wUMfOxyq+4eYabDWmf4Tr8yuXYqKisxh4ps3N0sbaAfVS/T3pBODdb1BYUq+s9Ccx+wv8A0qsTT21TQ163Ut3puG8r+amjoSP0j7J8jWY7npe5RtYydLQ0/KM5l1TSQwD+EKRk4B7AD7q8d6sV6sqkJu9pnQOkJCDIYUgLxzwSMHyrxNglBPbcO6TmLH6FZINpMUpr9IN4A2NxoeVx91txl1p5pLrLiHG1DKVIUCCO4iqA9Ku8dJdLRYW18GGlSnQPrKO6nPeAlX61V9pW3bQYsBu8aYjXxMVwndcghakrIODlKefEdYri6out4vF6fmX55124DDTpcbCFJKBu7pSAMEY48OeeusOHYKKaq6QSBwbftvotjF9onVdD0RiLC63YRrkcupenQNp+XdaWi1FJUiRKQHAP72Dlf7INbUrNHovWtMvXMq5LTlMCIooPYtZCR+zv1peozaSffqhH/SPE5+imdj6bo6IynVx8Bl53SlKVXlbEpSlESlKURKUpREqq9pW14aS1M7Y2LH66tltCnHVyejGVDeAA3T1Ecc1alZX9IYY2p3A9rTJ/o01jkcWjJV7aaunoqMSQOsS4DgcrHmpWfSCmdWmGB4zD/o1+D6QNy6tNxB4yVfuqlaVg6R3NUE7S4mf5vgPRXOrb/d/o6egjxeWa/Ctv18+jYrcPFaz99U3SnSO5rydo8T/unuHorgVt91F1WW1Dx6T/AEq/Ctvmp/o2izjxQ4f7dVFSnSO5r5/qHEv7x8PRWyrb3q36Nrsg8WnT/wCpX9Z29asD7ZettmU0FArShpxKinPEAlZwcdeDVS0r50jua8/r+Jf3itraXv1t1JZWLtanw7HdHHPBSFdaVDqI/wBuFdSszbEdotr0ZEnQLuxMcZkvJcbWwlKgg4wcgkd3Lsq99K620xqchuz3Zl5/d3iwvKHQBz9lWCcdoyK2WPDguk4RjUFdCzeeBIdRfj1BSKlKV7U2lKUoiVmz0itZbVdA6tSu26jUmxXAFcM+ox1dEofOaKi2SSOYJ5gjmQa0nUC27aEf2g6EXZoT0diezIRJirfyEbyQQQSASAUqV1HqqQwueKGoaZmgtORuL/NYp2ucw7pzWVVbdNqquerF+UKOP/Tr5q23bUlc9WyPKMyP7FQe922VZ7zNtE5KUyoUhyO8EnIC0KKVYPWMivHXQ20NIRcRt7gogyycyp+rbPtPVz1fM8m2x/Zr5q2wbTFc9YXHyKR91QSlevYqYfy29wXzpH8ypura1tJVz1ldfJ3H3V8lbU9oyues715SVCobSvvslOP4B3BOkfzUtVtM2hq561v/AJTnB99fj/dH2gbwV/DbUWR/1k7j+tUVpXr2aH+gdwXzfdzX+gWxy8zdQbMLBd7i8XpkiIOmcIAK1JJSVHHWcZqW1XXo1OdJsQ02rsaeT7n3B91WLXMKxoZUSNGgJ81Nxm7AepKUpWsvaUpSiJSlKIlKUoiVD9s1p+Wdmt5jJQFOtMest9oLZ3+HeQCPOphX5dbQ60ppxIUhaSlSTyIPMVkikMUjXjUG/csU8LZonRu0cCO9YRrW+xi/s3HZda5kqQ22YjZivrcWAEls7oyT+TunzrK+pLcq0ahuNqUcmHKcYz27qiM/CvjBamz3mLZES8+t50BphJJ3lqwOA7TwroeI0LMQhb71uN+qy5PhOJSYVUOO7vE5W67rr7SI8CNrq8N2yVHlQlSlOMuMLC0bq/awCOBxnHlUfrQ2jdg1sZityNUTX5MpQBVGjr3G0dxVzV4jFTeDsu0DDx0WmoqyP78tbv8AXUa0XbQUsAEYu62V+akWbK11S4yu3WAm9idL9gWQq8t2iiba5cNQGH2Ft8fykkffW5oOndPwMeo2O2RscuiioR9grMPpAxBE2q3QpTuofSy6keLaQfiDWagxhlfKYdywtz/OawYps/JhcLagvubgadpvr1LDZBBIIwRwIpXU1dFELVFzjAYSiU5uj8neJHwxSqHIwseWngunxSCSNrxxF+9XVsrY9X0JbgRhSwtw+a1EfDFbS9HKJ6tsshO4wZL7zv7ZR/YrH+jWuh0laG8YIhNE+JQCa27sqiGHs0sDCcJUYDbnLkVjf/tVbca/Z4dDH2eAVE2dHTYtPMf+Xi5Uxsh/+L+kBcbn84IdmSQewKUUD+vXu9LCXv3Wwwc/imHXSPz1JH9g1Ndk+zCRojUM26P3Zq4CRHLKSlkoUMrCiTxP1R11HNu+z/V2qtWNXOzwmZMZqGhhI9YQhWQpSjwUR9ascdXTvxNkgeN1rbAnLh19qyy0FXHgz4iw773XIGZ1HK/JWBsXiepbLbC1jG9G6b9dSl/2qyXfZfr97nzs59YkuO57d5RP31sF1KtPbNVIUAhVts5B48i2z/qrGVbGz/7SWebmfUrU2q/ZQU0HIfQD1WhPRQhFFjvlxKeD0ltkH8xJUf8AOCrsquPRyg+qbLobxGDLfefP624PggVY9VvFZOkrJHddu7JXDBIuiw+FvUD35/VKUpUepVKUpREpSlESlKURKy16RYxtQmHtjsn9gVqWsu+keMbTXz2xWT8KxTfCqptiP/Tx/wBh5FVvSlK1ly9KUpREpSlESlKURfWLHkSn0sRWHX3lnCW20FSleAHGtB7DNmc6wSm9TXpxTExTSkswwOKEqHNZ7cfRHLr48BBPRpdcb2k7iBlLsF1C+4ZSftArTtZ4mA5q97KYPBK0Vjzcg5DgCOPWlKUrOugJSlKIlKUoiwn6R1rVats2oGynCJDyZSD2hxAUf2ioeVV5WhvTU07JZ1FaNVNoJiyI3qTqgOCXEKUpOfzkqOPzDWea6fhUwmo43dVu7JQk7d2QhKUpW+sSUpSiJSlKItweiy5v7DrEn6ipKf8A6hw/fVn1WnowMdBsP0/kYU4JDh833MfDFWXXLcR/3ctv6j5qch/dt7EpSlaayJSlKIlKUoiUpSiJSlKIsj7eIQg7VLylIwh5SH09++2kn9rNdb0aIbMraWHXkhSosJ15vPUrKUZ9yzXQ9KeF0OtbfOSnCZMAJJ7VIWrPwUmuL6O1xbgbT4bbiglMxlyPknrI3gPMpA86vzXmXB7t/pt3ZfRcufG2DH7O03we83Hmr82qa6i6FsrMtyKqZKkuFuOwF7oOBkqJwcAcPePEUvP29aveyIsG0xU9RDS1qHmVY+FW3to0K9rewx2oMhpmfDcLjPS5CFhQwpJIzjkDnu76qGHsH1m8r8PJtMZPap9Sj5bqTURhLcMEG9Pbe439FO447GTVFtLfcsLW8bntXEn7XNoEvIN+Uwk/RZjtox5hOfjURvN1uV5mmbdZz82QUhPSPLKlYHIeFXNB9HqSrBnanZb7UsxCr4lQ+yovtk2bw9DW+2SYc+TM9accbdLqUgAgApwB+l11OUtbh3SiOntvHkLeNlW67DsXEDpqq+6Obr9WlysdbWmOg13OIGEuhtweaAD8QaV1ttcNxzVrDjachUJBPjvrH3ClUzEoi2rkAHEroeDztfQQkn+EeGStu0thq1RGhyQwhPuSK0npzbnpWLbIkCTa7qwI7KGgUIQtOEpA+sD1dlZyYTusNp7EgfCu1pzTV/1E6tuyWqTNKPnqbT7KfFR4D31eq2hp6iMdPkG9dlzLDsSq6SZxpsy7ha91pqDtj2fysBV4cjKPU9FcHxCSPjXega50dOwI2p7SonklUpKFHyUQaybqXSGptOIS5erNKiNKOA6oBSM9m8kkZ7s1zbVbp91nNwbbDflyXD7LTSCpR7+HV31EHZ6je3fjkNudwQp9u1mIRv6OWIX5WIP58lq/bBdIyNlV+kxpLTyVRw1vNrCh+EUlHV+dWRqsjUWyi56a0DK1He5bbcpCm0ohte1uhSwDvq5Z48hnxqt6kMFp4YInCJ+8L626h+XUVtFVVFTOx08e4d3S9+Jz6uxbD2ONdDswsCMYzECv1iT99S2uBs4bDOz7TzY6rZHJ8S2kmu/VCqDeZ56z5rqFILQMHUPJKUpWFbCUpSiJSlKIlKUoiVSW2vZtqXUur03eytR32FxkNrC3ghSVJJ7eYxj41dtK8uaHCxWjiOHxYhD0Mt7Xvkstp2La7POHDHjKTX0TsT1yebEAeMofurUFK8dC1QX+jsP5u7x6LMSdh+tzzFtHjJP+jUL1lpq46UvJtN0LBkBtLn4FZUnB5ccDsraNZi9JYY2k57YLR+Kq8SRhouFC7QbP0lBSdLDe9wMz2qsqUpWFUlKUpRFOthFzctm022BBSES96K5kZyFDhjs9oJrWFZI2UDS7F/Yueob89bHYUlp6MlDClpcKVZIUQDgcAPM1p/TeprDqNDyrJc2ZoYIDoRkFOeWQQD1Hj3VsQnKy6TsfM1tMY3PFybgXF9M8teC7FKUrMrklKUoiUpSiKNbTtKRta6IuWnpG6lUhvMdxQ/FPJ4oV78Z7iR11/n7cYcm3XCRAmsqZkxnVNPNq5oWk4IPgRX+k9ZJ9MPRptWrY2rIbBTEuqdySUp9lMhA5ns3k4PeUqNWjZqt3JDTu0dmO37jyWjWx3G+OCoalKVdlGrSWy70dbJf9G22/X6+XFLtwjpkIZhbiA2lQykEqSrJxjPAdlSk+jBoXHC86kHi+z/7VWBsMc6XZBpZXZbm0+4Y+6ppXOqrF60TPAkIsT5qXZTxloyVDq9F/Rf0b7qAeLjJ/9Ov416L2jQ6C5fr8pvrSFNAnz3Puq+aVh/Wa7+4V79ni5Ln6cs1v09YodltTPQwobQbZRvEkAdpPMk5JPfXQpSo1zi4knUrMBZKUpXxEpSlESlKURKUpREpSlEVD+lo37Om3QOOZKSf8lj76oyFJfhTGZkV1TT7DiXGnE80qScgjzFX96WDebNYnsfNkOp96Un7qz802t11DTad5a1BKR2k8q6DgJDqBoPX5lcp2oaW4m8jjbyC0toTbZp65wG2dRu/JVwSAFqKFKZcPakjO74Hl2mpRK2n6CjI33NTQ1D/BhTh9yQazevZjr1HPTMw+BSfsNfBezvXKOel7mfBkn7K0JMHw2R+82Ww5AhSkW0GMRRhr4bkcS13ir6nbcNCxs9C9cJmP7zFIz+uU1WO2XadadbWONbYFsmR1MSg+HXykZG6pJTgE/WB59VQxehNaI56UvR8Iaz9grwXPTuoLZHMm5WK6QmAQkuSIjjacnkMqAFblJhVBBIHxuu4aZ+ij6/G8TqYnRyts06+6fqq81rbETbq06oDIYCf2lfvpUguUfpX0qxyTj4mlZ56Jr5C4jVa9NiL44msB0XuQd5AI6xmtqaCs0TT+jrbbozaG0tx0KdIGN9wpBUo95OaxPCVvQ2VdraT8K2BtNde/3H7ouMVBa7ekDd54Vugj3E1H7Qgy9DEDYOPp6qV2ULYfaJiLlo9SfJSaYxa9QWiRDdMedBkJU06EqC0ntGR1j4GuNs40XbNGWRMOI2hyUvjJlEe26rsz2DkB58yaq30VFzWZt8hvNvIYW006kLSQN4FQyM9x+Arq7bdcauhXd3Tml7fMaQhtJfmssKWtRUM7qCBhOARx555YxUK6gmbUOoo3+7kTwH5mrEzFKd9KzEpo/ezAAzOug7vlmul6Rt/szOhJticns/KchTSmoyTvLwHEqJIHzRgHnjPVWY66lws2oUoeuE+1XRKM77sh+O5jJPNSiO09dcurfhlGyjh6Nrt7O5PWqBjOISV9R0r2buVgOrNbY0Tu/wADLJufN+To+PDo012Kj+zd0PbPdOr/AOrI4PiG0g/ZUgrnMwtI4dZXXad29E08wEpSlY1lSlKURKUpREpSlESlKURKUpRErM3pNDG0Zo9tvaP7S60zWafSeGNocY9ttbP9I5WKX4VV9rx/6d//AEFVdKUrWXLEpSlESurpfUN201dUXOzy1R30jdV1pcTnilQ6xwrlUpovccjo3B7DYjitk6N1TC1BoyPqRSkRWVNKVIC18GVIyF5PYME57MGo3ss1uvV+rNThDpVAjlkQE7uPwfthSsdqjg8e0Dqqg7Fra4WnQ930q00lce4qCukKiC1nAXgde8ABU19FlwjWVzZzwVbyojwcQPvrOJLkBX+k2ifWVVLED/26zYju4/4XIVtdumgtueobXcXnJWm3rmvpmFZUqNvYy432c8lPI8evjU92l+kBpKx2Nz+C81i+XZz2WUISsNNflrUQMgfVBye7nWcfSATu7ZdTj/pmfelNQSujMwWlqWxTPGdhcDQ5cVZDUvYXNC3P6P2ub9r/AEnJvN7gQ4vRSfV2lRwoB0pSCpWFE4GVAeRqyKzb6K21OwxLFF0HeVN2+S26sw5KyA3I31FW4o9S8kgZ4EYHPge/tY9IKHpLVqbDZbY3dTEdAuLqnd1I4cW2yPpDrJ4AgjB6q1V4XO+tfFFHbiOVud/zNbkc7RGHOKvOss+kLtutV+tk7R2nbezOiOK3JE+QnKcpOcsp7iPnnyHXXI2oekTftQMP2zS8Y2W3OpKFvLIXJcSRg8eSPLJ/KqjamcHwIxO6apGY0HLrK16iq3husSlKVa1oLd3o6OdLsV00rsjrT7nVj7qsGqw9Ft8PbD7EnraVIQf8u4fsIqz65ZXi1VKP+R81ORfu29iUpStRZEpSlESlKURKUpREpSlESlKURKUpRFTnpV7n8EbVn5/r/Dw6NWfuqgNPJ37/AG5H1pTQ/bFXr6WDmLPYmfrSHVe5KR99UdpVSU6ntSlnCBNZKj3b4q+4ECMPv2rl+0xBxW3/AFW3HHENIK3FpQgc1KOAK/CJEdfzH2leCwajG1+3S7rs2vcKC0p6QtgKQ2kZUrdWlRAHWcJPCsdqBSopUCCDgg9VV7C8HbXxl+/Yg2ta/wBQrZjWPuwyVrOj3gRe97cdNCt4VWHpMK3dmZH1prQ/rH7qp7YBaply2lW55htwxoRU/IWPmoASQnPiogY/dVs+lG6EbPIqCeLlybAHg24furI2gFFiMUQdvaHS31KwPxQ4jhE8xZuixGt7+A5rL77yW1gEdWaVyr/JDMxCSoDLYPxNKuElQGuIVDipC9gcvZpxzptPW17nvxGle9ANbw04rf09bV/WiNH9gVgHZ6+JGirS4DyjpR+r7P3VPW9bawbYQw1qe7ttNpCEJRLWkJAGABg1GV2HuxGCItcBYX7wFL4ZirMIqpw9pNzbLqJW0aVitesdXL+fqm+K8bg7/pV8F6l1Gv59/uqvGY4fvqLGy8nGQdymjtrDwiPeFqnbWnf2WX4f9HB9y0msg17JF2ukltTci5THkK4KS4+pQPiCa8dT+FYeaGIsLr3N1VsbxVuJzNlDd2wtrfiSthbG3em2YWBec4ihP6pI+6pdVb+jhO9b2XRWScmHIeY/a3/7dWRVBrWblTI3kT5rqWHSCSkieOLR5JSlK1VupSlKIlKUoiUpSiJSlKIlKUoiVW21bZcNb3mNdGrx6g61HEdSVMdIFJClKB+cMH2j8KsmlfCARYrVq6OGsiMUwu385Kh0+j479LViB4QCf/Ur9p9HsfS1cfK3f/61etK8dE3kor/S+F/2v/s71VHD0fGOvVbh8II/9yq02q6ORonUTFqbnqmh2ImR0imtzGVrTjGT9X41rys3elIP9/kBXba0D+ldrxIxobcKD2iwWho6IywMs644k+ZVTUpSsC5+lWt6L693aDLT9a2OD+kaP3VVNWl6Mn8or3+Lnf66K9M+IKWwI2xGHtVT+kSnd206mH/SUn+jRUAqw/SRTu7bdSD/AAzR/oW6ryuwUP8Ato/+o8l0iX4z2pSlK2l4Wh9gGxzSGuNAJvl6XchKMt1khh8IRupxjhunjx7asVPo37NxzTd1eMz/AO2vx6Has7IlDsubw/ZRVzVz7EsRq46qRjZCACeKloYYzGCQqgT6OezUc41zV4zD+6von0d9mQ526erxmrq26VofqdZ/dd3rL0Ef9IXK0rp+0aXsUeyWOIIsGPno298qOSSSSVEkkknnXVpStNzi8lzjclZALZBKUpXlfUpSlESlKURKUpREpSlESlKURKUpRFQ/paOD/e20Dx/jKiP8kB99UQlRSoKSSFA5BHVVt+lNND2uIEJKsiNASSOxSlqJ+ATVR10bBGbtDGD1+JK5HtJJv4nKR1DuAWw9l+tIGstOsyWnkC4NICZsfPtIXyKsfVPMHy5g125thsU54vTbLbpLpOSt6Khaj5kVim3Tptulol2+W/EkI+a6y4UKHgRxqYRNre0KM2G0ahWtIGB0sdpZ95Tmoap2ckEhdTPAB4G4t3KwUe10JiDKuMkjiLG/XY2WsIcSHBY6GHFYitA53GmwhPjgVnL0jtaQ7/dotjtT6X4lvUpTzqDlK3jwwO0JAPHtUeyoVqDX+sb8wqPc7/LdYWMLabw0hQ7ClAAI8ajNbmGYGaaXppnXcNLLQxraVtZD7PTs3WnW+vZYaKstq129Q1CwzvYzESr9tY+6lRvbQ+HdaqbB/Ex20H4q/tUqAxKtkbVyBpyBVqwfDonUMTnDMgFT7Y1I6bQ7Lec9A8434cd7+1UzqsdgsvehXSCT8xxDqR+cCD/VFWdVwwiTpKKM9Vu7Jc/x6LosRlb1378/qlKUqRUQlKUoi0H6J80rs18txVwZkNPgfnpKT/mxV21mb0YLoIevH7cs4TPiKSgdq0HfH7IXWma55jsXR1r+ux8PVdZ2ZmEuGs5i48fSyUpSodT6UpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJWcvSlH+/W2q7bckf0jlaNrOvpTj/fdaldsDH9IqscvwqtbWD/013aPNU/SlK1VylKtH0ZP5RXv8Xu/10VV1Wj6Mn8orv+L3f6yK9M+IKVwP/wDIw/8AYKsfSaTu7cdRj8tg++O3Vb1ZnpQjG3TUPf6sf/pmqrOuv4f/ALSL/qPILpUv7x3aUpSlbaxrYvobKzsmkDsuzw/o2quqqQ9DA52VTh2Xl0f0LNXfXMsW/wB7J2qap/3bUpSlRyzJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKV+H3W2WVvOrCG20lS1HkABkmiLJO3WaJ21O9LScpaWhgd24hKT8QahFe7UNwVdr/cLosYVMkuPkdm8onHxrw11Ski6KBjOQA8FxGum6epklHEk+KUpSs61UpSvytSUIUtRwlIyT2CiLPe0eR6zri6uZzh7o/1AE/dSuNcZKplwky1fOfdU4fFRJ++lcpqJOllc/mSe9dxpYuhgZHyAHcFMtic0R9XKiqPCVHUgD8pOFD4BVXbWbNJzvk3UtumlW6lqQgrP5JOFfAmtJ1c9mpt6mdH/AEnz/CuebY0+5VtlH8Q8R9rJSlKsaqKUpSiLs6Gu3yFrG03YqKURpSFOY+pnCx+qTW1qwhWyNlN6Tf8AZ/aJ4XvOiOll7jx6RHsqz4kZ8xVS2og+CYdn1H1V82Lqf3tOepw8j9FKKUpVRV8SlKURKUpREpSlESlKURKUpREpSlESlKURKg203Zvb9cSYkqRcJEJ+MgthTaAoKSTniD2HPvqc0r4QDkVr1NLFVRmKZt2ngqXT6P8AavpaimnwYSPvr9p2AWT6V+uB8G0CrlpXno28lGDZzDB/KHefVU8nYDp/6V7uh8A2PuqT7PtmNj0ZdHbnBlTZMlxks5fUndSkkE4AA48BU6pX0MaOCzwYJQQPEkcQBGmqqjX+wnSutNWTNSXO53piVLDYW3GdaSgbiEoGN5snkkddcVPoxaBHO7alV/8A2Wf/AGqvGlSTMUrGNDGyEALfMEZNyFSSfRl2fDnP1CrxlNf+3X0T6NWzsc376rxlo/0Kumlff1at/uFfPZ4/6VHtA6Psmh7ALJYWXW43SqeWXXCta1qABUT4ADh2VIaUrRe90ji5xuSsoAAsEpSleF9SlKURKUpREpSlESlKURKUpREpSlESlKURKhm2q7/I+zO8PpWEuvs+qt9pLh3TjwSVHyqZ1RfpWXpKY1o082v2lLVMeT2AAoR78r91b+GQdPVsZ1+AzUZjNT7NQyycbWHacgqDpSldMXGkpSlESuLriaLfpG6Ss4IjqQk/lK9kfEiu1Vf7cZ3QaajwkqwqVIBI7UoGT8SmtPEJuhpZH8gfspDCaf2itij5kdwzPgqYpSlcwXaErR2ibj8q6Ut00q3lqZCXD+Wn2VfEGs41bmwq6dJb5toWr2mVh5sH6quB9xA/WqwbOVHR1RjOjh4jP1VV2upelohKNWHwOR+isulKVe1zFKUpREq8/RW1BuSbnpl5fBwCXHBP0hhKx5jcP6JqjK7egr6vTesLZeklW5HfBdA5qbPsrHmkmtDE6X2mlfGNeHaFJ4NW+x1rJTpex7DkfVbVpX4YdbfZQ8ytLjbiQpC0nIUCMgiv3XM12VKUpREpSlESoa3rlc5Ti7Bpi8XeKhxTYltBttpwg4O4VKBUM8M4qVXJRTbpKknBDKyD2cDUV2VO+rbKbS+ll17o4al9E0nK1kFRwkdZPV30Re/T2rI90uzlnlW24Wm5Ia6YR5jYHSN5wVIUkkKAPCvI/rbpp0qLY9PXa8iI6WXn2EoQyHB85IUpQyR14FfK26ktV01jBjytNXi33Ux3TGenR0I/BjG+AQsns6q/mxfjoVpZ4qVMlFR7T0y6+ovdZNXNTryiy3C03Gz3B1tTjLctCd15KfnbikkgkdYrzS9bqReZ9sg6ZvVyVAdDTzsZDZRvFIUOageRr4639naBohQ4K9ZlDPcWONciyzdRxdaauTZLHFuLap7ZcU7N6EpPRJ4AbpzRFJbDrGPcr2LLKtF1tM5TJebbmspSHUg4O6Qo5xmvVH1NDfj359LD4TZXHG3wQMrKEBZ3ePYevFRSJKu8narZ3NT2xFqc9TkItyGXg+h5eAXN5fAjCQMDd86/lq/4K2lf97lf+XFEXTh67lzIjMuLonUbrD7aXGlpbawpKhkEe31g17p+sBb9MG9z7FdIx9YTHTEcSgPKKiACBvYwSe2uFou661b0fZW4mk4L8dNvYDTqrqEFaA2nCincOMjjjqr+7TJdzXs/ZlXa2CJKRcoxMZh7pyQHRjBwMk9lEXRd1xMYbU7I0PqdDSBlSksNqIHbgLr13jW1rgaSialZZlTokxaEMoYQOkUV5AGCRxyMY7a59y2gLjQH3xorVnsNlX4SAEpHDmo7xwO04NRtEQQtkui44kNSP/i0NfSNKyk7zylYz3Zx5URWfY7nEvNoi3SA50kaS2HEHr49R7CDkEdorwW/U9ulW+7z3OkjRrTKfjSFu45tfOUMZ4ceHX3VwrV/vP1suzq9iy3xxT8A/RYk83Gu4K+ckduQKi9xUU7L9ohScH5dlj3uN0RS9jWtwmMolW/RN+kRHBvNuqDTZWnqUEqXnB5iuxpXUcPUDckMsSokqI4GpUWU3uOtKIyMjJ4EciOddSEAmEwkDADaQB5VENNcNrGrgOAVHgk953FCviKWXSW3b7ZKnupUpuMyt5SU8yEpJIHfwqJQ9eSpcRmZG0TqR2O82lxtxDTRCkkZBHt9YqQay/8A0hef+4P/AObVUO0brJUTRtmjjSOqpHQ29hHSMQUqQ5htI3knf4g8wa+opHbtY22dpq53tqPNaFrS763FfaCH21Np3ikpzjOO/Fc+NrmbJjNSWNEajcZdQFtrDbWFJIyD8/sqOWV/5R0JtDvRR0Bm+tkxV/jWN1gp3XB9FXXiuvpe663Rpm1oj6RgvMphshtxV2CStO4MEjc4ZHVRF3ZOrI8HTYvN1ttxgFTwZbiONhT7qycJSlKSQSerjXiOsLulPSq0HqDocZynoivH5m/nPdXO1i/cZLmi1XWC3CkqvyCtht7pUpwle6d7Azw48uupXqK9Ls6GVIst2ufSkjEBlLhRjHzsqGM54eBoi8D+tLQjRv8AClgPyIW+hBQlIDiVKcDZSQSMEE8a9Gp9TRbHIiQ/VJlwnzN7oIkRsLcUE/OUckAJGeZNQfVVwtFy2R3WTZra/bmvlNtLzL6AhYeEhvfJAURz76kLxzttjpPEJ06sgdhMhIoiSdcyreyqVedH3yDCR+MkYbdS2PrKCVEgeVe3WWtLbpmFAmSGJMxiaT0aoqQrCQneK+JHDd48K9+twFaLvgIyDbpGf8mqoMlKX7fsuQ8kOJW2kLChkKBicQaIrMivsyozUmO4l1l1AW2tJyFJIyCPKvBpW9x9RWGPeIrTrTL5WEodxvDdWpBzgkc0mo5ota9NagkaJlLUYqgqVZnFHO8yTlbOe1BPDrwa8ezq6s2TYqzdn8bkVuU5g/SIfcwnzOB518RdqPre2vaxVptMaWFh1bCZZSOgU8hAWpsHOd4A9ldXUt6j2KEzKktOuIdktRwG8ZCnFboPE8uNVq+1BibK4spu729y/wAN8XpX8ZRvLkb2+tJGeJ3SU468CpHtGnMXPRdmuMVW8xJuMF1s/kqcSR9tfUU7pSlfESlKURKUpREpSlESsbbVL/8Awl13c7mhe9H6Xoo/Z0SPZSR443vM1pbbRqEac2e3GShwIlSU+qxuPHfWMEjvCd5XlWQqtuzNL8c57B9foqHtnW/BStP/ACPkPr4JSlKtyoaUpSiJVJ7a7j61qtEJKsohshJH5avaPwKfdV0SHW2GHH3VBLbaStaj1ADJNZovc5dzu8u4OZCpDynMdgJ4DyHCq3tLUbkDYh/EfAfeyuGx1L0lU6c6NHiftdeOlKVSF0hKkeza6i0awhPrVusvK6B3s3VcOPgcHyqOUrLBK6GRsjdQbrDUwNqInRO0cCO9anpXC0Hd/lvS0OapW88EdG9276eBPnz867tdSilbKwPboRdcSnhdBK6J+rTbuSlKVkWJKUpRFpv0btVfLOkVWSU7vTLVhCcnipg/MP6PFPcAntq1axtsu1OvSWtIV1KlCMVdDLSPpNK4K8ccFDvSK2M04260h1paVtrSFJUk5CgeRB7K59jtF7NUlzfhdmPr+da6tsziPtlGGuPvMyPZwP5yX7pSvy64hptTrq0oQgFSlKOAkDmSeyoVWJfqlZ22i+kb6vNdgaKgMSG0EpM+WCUqPahAI4d5PHs7a8Xt52mKd3xeY6E/UEFrd+Kc/GpWLBqmRu9kO1a7qqMGy2UtKVoKVDKSMEdoqCWO2a40tbkWa2RrNdbdHUoRXHpC2XggqJCVjdIJGcZFVfsp29alvWpYGnrxY49wXNeDSHoeWnEZ5qUkkpIAyTjd4A1o2tOppZKZ27IsscjZBcKJWSzX6VqpGo9RmAy5HiqjRYsNSlhIUQVLUpQGTwxgDGK8NutOsdMKlw7GxaLla3JLj7CZDy2XWd9RUUHCSCATwPOuJ6Q20ydoCDbGLKiK5cprilkSEFaUMpGCcAjiVEY8DVT2X0jNYfLEP5VjWkwOnR6z0UdYX0eRvbp3zxxnHCs8GGTzx9I0ZLw+djHbpV+QLNqS6aqt981J8mxWrYh31WLDWpwqW4ndKlqUByHICvM3a9aWnUl8m2eNZJEW5SUvJ9ZkOJWnCAnGEpI6qnLTiHWkutrStCwFJUk5BB5EV+qj1mUMg2TU1y1ZbL5qRdqjt2tD3qzEFTiypbid0lalAcAOWK+kLTdwZg6wZWuPvXl95yNhRwAtoIG9w4cR1ZqX1nLbBtq1hpTaNddP2tm1KiRC0Gy8wpS/aaQs5IUOtR6q2aWlkqXlkeuq8SSCMXKtGxsbRrVZYNrbt+mnEQ47bCVqlvZUEJCQT7HdXrv9p1JqDSzUS4NW2PPRPZeIZeWpvo0LSrmU53uB4Yrq6Cusm+aJst5mhsSZsFp90NjCd5SQTgdQ4199XXqPpzTFyvsrHRQo63inON4geynxJwPOsJY7f3ON7L1cWuumsBSCkjIIxVfwdGXZnZ9p2wLdiGXbbi1JeUFq3ChLylkJOMk4I6hVD/3Rmv8A+8WT/wAKv/TrQWxLWrmu9Cs3aWGUXBp1bEtDQISFg5BAJJwUlJ8c1uVOHT0zN9+ixMnY82CkGsbE1qKwP25ayy7wcjPj5zLyeKFjwPwzUb05o65r0RfbLqORG9bu8t+Qt2MSpKS4E8cEDkoZxU8rO22XbRq/SO0a5aftTVrVEjBktl5hSl+00hZyQodaj1VhpaWSpfuR66r3JIGC5VqxHNo8KK1DXbdPzi0kIEgS3G98DgCUlBwe3Fe7RljucG4Xa9Xx+K5c7otvpERQeiaQ2kpQlJVxPM5NRvYHtHXr/T0n5STHavEJzEhtkFKVIVkoWkEk9oPHmO8VZVY5onwvLH6hfWuDhcLw6giOz7DcILJSHZEV1pBUcDeUkgZ7uNfLSsF616YtVsklBeiQ2WHCg5SVJQEnHdkVwdsGtGdC6JlXjDa5qz0MJpfJbquWR2AAqPcMddUfobbzre9a0slnmM2cRps9iO6W4ygrdWsJODv8Dg1sQUE08ZkboF4fM1jt0q7Bpa4iFrZgLjA3zpDF9s4TvM7nt8OHHszXztKNo9vtcSAi3aaWiMwhlKjLeyoJSBk+x3VOaju0nUjekdD3W/q3C5FYPQIVyW6r2UA928RnuzWqxpe4NGpWQkAXK8V9suob/p+GuUu3QL5AnImRi0pTjBUjIAVkBWCCc4r+et7RyC2LLp1K8Y6UznSjPbu7mazr/dGa/wD7xZP/AAq/9OtH7JtVjWmg7dfVhtMlxJblIb4BDqThWB1A8wOwituqw+amaHP0WKOZshsFzp2ip3+5q9p1iUw/cZElMp95eUIU4X0uLxgEgYGB4CulqqyXleoYOpNPOwvXo7C4zrEveDbzSiFY3kglJBGeVSqs0bTduWs9Oa9vFjt7NpMWHILbRdjqUrGBzIWO2sdLSSVTi2PgvUkjYxcq3buxtCvlsk2l2FYrYxLbUy9IElx5SUKGFbqd0ccE8zXQuOmXvXNJpgLbESyOEL6RRCigNdGnGBxPLsrOKPSO18lQJi2JY7DFcwfc5Uy0N6STEqa1E1dZ24aHDgzIalFCD2qbVk47wSe6tqTB6pgva/YsYqYybXVy66sL17tjTlveRHu0F0SYD6uSHB9FWPoqHAj7cVHIGi7udDWDS85cMsMTi/cwlZKXWg6twNpyOOSU5zjlVgR3mZMduRHdQ6y6gLbcQrKVJIyCCOYIqstsW2O0aEeNqix/lO9lIUWArdbZB5FxXaRxCRxxzxkZ0YYZJn7jBcrM5waLlTb+CGk/+a9k/wDANf6NRV/R1+a0R/B+G5CcVCuokW8uuqCfV0ub6UrO7kKGSOGeGKz/AHL0gNo8t5S482DASTwQxDQoDu9veNeywekTrqC+k3Nu3XVnPtJWz0S8dykYA8wakjglSBfLvWD2qNaJ9Y2k/wD7Xpj/AMW9/oVLI5dMdsyEoS8UDpAg5SFY447s1zdIXdy/6Yt16cgPW9U1hLwjuqBUgHlxHaMHwPIcq+Wu7+zpfR90v7+6RDjqWhKuS18kJ81FI86iujcX7nG9lsXFrrt0rIv90Zr/APvFk/8ACr/060Rsb1grXGg4d7fDSZoUpmWhoYSl1J6hk4ykpV51uVOHTUzN9+ixRztkNgplSlK0FmSlKi21LU6NJaMm3QKSJRT0MRJ+k6r5vjjio9yTXuON0rwxupyWOaVkMbpHmwAuVQ/pGaq+XNZfJEZzehWnLXA8FPH8YfLAT+ie2qwr+uLW44pxxRWtRKlKJySTzNfyuoUlM2mhbE3guLV9W+sqHzv4nw4D5BKUpWwtRKUpRFD9rt1Fu0e8whWHpqgwnt3TxV8BjzqiKm+2S7/KGqfUm15ZgI6Pu3zxUfsH6NQiueY5Ve0VbraNy9fFdZ2aovZaBt9Xe8fnp4JSlKh1PpSlKIrE2JXv1W7vWV5eGpY32s9TiRx96f6oq46y9BkvQprMuOvceZcDiFdhByK0jp66MXmyxblH+Y+gEpz81XIp8jkVdtm6zpIjA45t07PsfNc42vw/opxVNGTsj2j1HkuhSlKsqpyUpSiJWjPRs1r8o2pWlLg9mXCRvRFKPFxnrT4p+wjsrOde7T92m2K9RLvb3OjkxXA4g9R7Qe4jII7DWhiVCKyAx8dR2qVwfEnYfVCX+HQjq+2q3FVPelhqN+zbOkWuKtSHbvIDC1JOCGkjeWPP2R4E1ZOjdQQdUadiXqAodG+j20ZyWlj5yD3g/ceuqO9NNLph6WWM9EHJQV+cQ1j7DVDw+L/zGseND5LrckgfDvsNwR5qBejXoOBrTVsqReGQ/bLW0lx1kkgOuLJCEnH0fZUT4Adda7jW23RonqkeBFZj4x0TbKUo9wGKz/6Fj7XRaojYAd3oy89ak/hR8D9taLrNjEr3VJaTkLW7kpmgRgqPwdF6Vgal/hHBscOLdC2psvMo3MhWMndHs54fOxniePGpBSohti1MNJbOrtd0Obkrouhi8ePTL9lJHhne8Emo5ofM8NvcnJZjZoJWUtvepFas2pXF6OouxoqxBiBPHKUEgkdu8sqI8RX7226AVoSbZG0pPRzba2p1Wcj1lAAeA7slJ/SrhbMJVjg6+tNx1I+pq2xHxIdIbLhUpHtIGBzyoJz3Zq3fSE2h6A1zopuNari+5dIclL0cLirTvA+ytOSMDgc/oire4yQSxRRtO6BY5dyjRZ7XOJzVm+jZqf8AhHsvhNPOb0u1n1J7J4kJA6M/qFIz2g1ZlZH9E7U/yPtBcsj7m7GvLPRgE8A8jKkHzG+nxUK1xVbxSn6CocBocx81vU799gSsUekl/LXqH85j/wAu3W16xR6SX8teofzmP/Lt1t4D/uHdn1Cx1fwDtWrdkP8AJZpf/FUf/Niqv9MLU/qenLdpWO5h24OesSAD/NNn2Qe4r4/oVaGyH+SzS/8AiqP/AJsVkPbfqf8AhZtKutxbc34jTnqsUg5HRN8AR3KO8r9KmHU/S1rnHRpJ8cknfuxAc179M7PXLrsY1BrIIUX4Upv1YD6TSB+GPh7YOf8ABmpP6I2p/kzW8nTr7mI92Zy0CeAebBUPenf8SBU20HtQ2V2LZtA0nKuMhxCYRZlpEJzC1uAl3q5EqV5VnG33BVh1Szc7RILvqEwOxnSkp6QIXlJI5jIAyO+pVokq2TRSNIB0v4eV1rHdjLXNPav9CKxb6Tf8td8/Njf+XbrYlguka92ODd4St6PNYQ+34KAOD3jOKx36Tf8ALXfPzY3/AJduorAwRUuB5HzC2avOMLibJNXyNDa5hXg7/qqsNTWwPnsLxnh1kcFDvArdEV9mVGakx3UusuoC21pOQpJGQQewisf7U9FlnZpo/W8Fr8G/bmI0/dHJYT+DWfEDdP5qe2uppba69a9g9w08ZBF5YcEKCvPtCO4FEqH5gCwD1ZRW3X03tobLFrex7/p5LFDJ0V2u7VxPSL1urWWu1w4LpctdsJjRQniHV59twduSAB3JHbUV2Ufyn6W/xvF/zqamHo56LOoL7Ov8xret9mYU4nI4LkFJ3B+jgq8QntqH7KP5T9Lf43i/51NSTOjZG+Bn8I8wVhNy4PPEre1Zx9MfU/8AwTpGO5/06UAfFLYP7Zx+bWjHVoabU64tKEIBUpSjgADmTWCdpmo16t13dr6pR6KTIIYB+i0n2UD9UDPfmq/gtP0k++dG+a3Kp+6y3NSFrZ64vYS5rro1+sC4gAdXqo/BlXj0h9wqeeh3qf1e73PSchzDctHrcYE8OkTwWB3lOD+hUmjbTtkrezlGil3WSYnyd6ktQguccowV8uecq8aznoq+vaX1jbb7GUVmDJS4d3h0iM4Un9JJI86lg2WshljkaRnlfwWtdsbmlp7V/oDWGduv8rupf++H7BW4IUliZDZmRnA6w+2lxpY5KSoZBHkaw/t1/ld1L/3w/YKjsBymd2fVZ6v4Atc6LsNilaEsYlWW3Phy2xysORUK3iWk5zkcayn6Q9gs2nNp8y32NpDEZTTbymEfNZWoZKR2DkcdW9X6d2o7VrXZ4cF28TYMLoEtxd6A01vNpSAN1fRgnAxxz2V59megr/tQ1BJe+UEBttxK7hMkvb7o3ieISTvKUcHieHaRW/SUr6N7ppXjdWGSQSANaM1obYbfXrd6PLF7uAU6i2xpbiMnittpSyE/DdHgKyza41y1rrhiM7I3594nALeXxwpauKj3DJPlWwNoNkjWDYTebHaEFEaFZ3G0DmSkJyonvPEk9pNZY2EvtR9r2mnHgCkzAgZ+spJSn4kVjw54LZ52DO5t5r1MDdjCtjaO0bpzSVsZg2W2MMBse08UAuuHrUpfMk//AIwK/OpdEaS1IhKb1YIMpSVbyXOj3XAfzk4OO7OD11IqVW+mk3t/eN+a3d0WtZfxKQlISkAJAwABwFZ99MXU/Q2y16SjuYXJV65KAP0E5S2D3FW8f0BWgiQASSABzJrCO17Ux1btDu15QsqjKe6KL2dCj2UEeIG94k1J4NT9LUb50bn8+CwVT91lua7Fp2euTNh101uG1esMT0dEO2OnKXCB+csf5M1MfRA1P6jqmfpeQ5hm5NdNHBP882OIHijJ/QFS/TW0/ZPbdm8XRz9zkrji3mLIxCcwsrSekPLrUpR86zhp27O6b1XCvFvd6VUCUl1tQBSHEpVyxzAUOGOw1MNElXHLHI0jPK/h5LWO7G5rmntX+g1K8tpnRrpa4tyhr6SNLZQ80rtSoAg+416qqJBBsVJL+HgMmsobctafwt1WpmG7vWq3ktRsHg4r6TnmRgdwHaatv0h9bCw6e+QIDwFyuSCFlJ4tMclHuKvmj9LsrMlW3ZzD/wD3Lx2fU/TvVC2uxX/2cZ63fQfU/JKUpVtVESlKURK5mqLq3ZLDLuTmCWW/YSfpLPBI95FdOqi23X0PzWLCwvKI+HZGDzWR7I8gc/pd1aGJ1gpKZ0nHQdv5mpTBqA11W2Lhqewflvmq4kOuPvuPvLK3HFFa1HmSTkmvxSlczJvmV2MAAWCUpSi+pSlKIlWRsTv4jT3bDJcw1JPSR8nk4BxHmB8O+q3r6R3nY8huQwstutqC0KHNJByDW1RVTqSdsreHlxWjiNEyupnQO46dR4FajpXG0ZfGtQ2Bi4I3Q7jcfQPoODmPvHcRXZrp0UjZWB7DcFcamhfDI6N4sQbFKUpXtYkpSlEU+2L68c0bf+hlrUqzzFBMpHPozyDoHaOvtHeBV07f9Jr1xs2WLSEyZkRSZsPoznpgEnKQeveSokdpCayxV5ejztFRH6PSF8kBLalYt76zwST/ADRPYfo+7sqt4zQOa4VkA95uo59fr1K6bMYyG/8AhTn3T8J5Hl8+HX2qi9mWs7loDViLzDZDwCVMyYy1FIdbPNJPUQQCD1EdfKtL230hNnkmKHZL1xhO7oJacilRz2AoyD8K6uv9jWi9YTHLhIjP264OHeckwlhBcPapJBST34BPbUCV6MVt6XKdXSw39Uw0k+/e+6omapoKyz5btd+dqu7WTRZNzCkWnNvum79rqFp+LAlMQ5ZLSJslSUfhT8xO4M8CeGSeZHCoL6Yup/WLva9Jx3MtxEetyQDw6RWQgHvCcn9OrM0TsO0NpqS3MXGfu8xshSHJygpKFdoQAE+8HFePV2wfTep9STr9cb3fPWprpcWEONbqeoJGUE4AAA7hWCGahiqQ9lwAO8r25srmWOqqnYrsUh640iq/3W5zIKXJC246GUJIWhOAVHI+tvD9Gpz/AHM2nP8AnHdv1G/3VcmlLHC01pyBYrdv+qwmQ0grxvK7VHGBkkknvNdSsM2K1DpCWOsOC9Np2AC4zWAbxEuGitdvxUOFE2zz/wAE5jGS2vKF+BwD51rrVu1ixaf2d23ViketOXRpC4cNLgStxRAKgTg4CeIJxzwOuvJtC2KaY1pqZ2/zptziSnm0IdTGW2EqKRgKO8knOABz6q9UfY7pD+BbOlriiTco8Zbiosl9SRIj75yQhaUjA3snBBGTxzWzVVtLUiN0l7jX6+K8RxSRlwauTF9ITZ47bfWXnriw/u5MVUUleewEez8RWYteXx/W2v7heWIq0uXGSAwwPaVjAQhPecBI8avub6MtjW6DC1PcWW88UvMIcOPEbv2VNdm+xvSWipqLmymRcrmj5kmWQeiOMEoSAAPE5PfWSGqoaO74blxXl0c0lg7RfHX1zXs52CNxy4lE9i2s21gpP8+WwgqB7gFK/RrMex3RR15rVmyLfdjxQyt+S82AVIQkYGM8OKikeda32pbPLbtCiwYt1uVxisQ1qcSiKpAC1EAZVvJPIA4/ONefZbsusGz1+dItUmbKemJQhS5SkEoSkk4TupHMnj4CsFNXx09M+x/aO/PuvckLnvF9AoF/czac/wCcd2/Ub/dVU7d9mDWzuTbHIM2RNhTULBceSAUOJIyOHDBBGPA1s6ovtK0Tade2BFnu7khltt9L7bscpC0KAI4EgjBCiOVeKXFpmygyuu3ivslO0tO6M1XPoi6n+UtFStOPuZftL280CeJZcJI9yt/3iqb9Jv8Alrvn5sb/AMu3Widm2x+x6D1Aq82i73d1xbCmHGn1tlC0kg8QEA8CAedeXXuxDTWsdVS9RXC6XdiTKCAtDC2wgbiEoGMoJ5JHXWeGtp4q18wPukcuOS8uie6IN4hdHZ/ZIWo9gtnsdxRvRploQ0vtTkcFDvBwR3gVlG77OtaW69SrZ/Bq7SVMPqaDzMJxTbmDgKSoDBB55rbulrNH09p2BY4jjrrEFhLLa3SCtQHWcADPlXTrVp8TdTPeWi4Juvb4A8C/BQnQWkWdFbLhZEBJkJiuOy3E/TeUnKj3gcEjuSKx/so/lP0t/jeL/nU1vGUymRFdjrJCXUFBI5gEYqo9O+j7pSx3+33mNd704/BktyW0OONbqlIUFAHCAcZFZaGvZG2UynN33XmWEuLd3gur6Sep/wCDmy+a0y5uy7ofUmcHiAoHpD+oFDPaRWbNiOz9O0LVD9ukSXokKNGLzzzSQVA5ASkZ4ZJJPgDWpNqWzG07QpEFy73S5xkQkrS01GWgJyojKjvJPHgB5V6NluzmybPYs5m0Py5Cpq0qddkqSVYSCEpG6kcBknzpTV0dNSFrD75/PJHxOfICdFXX9zNpz/nHdv1G/wB1U7ty2dp2eX+FEiyn5cGZH6Rp51ICt9KsLTw4cPZP6VbaqH7UNn1m2g22JCu70qOYjxdadjFIWMjBT7QIweB/RFfKTFpWygzOu1fZKZpb7ozUW9FnU/y7s1btr7m9LsznqysniWjxbPhjKf0Kzlt1/ld1L/3w/YK1Lsv2VWfZ9c5c20XW6yPWmQ06zJW2UHByFeykHI4jn9I1xdXbBdL6l1LOv0y63lqRNd6VxDS2whJxjhlBPV21mpq2nhq3yA+6epeXxPdGG8Qule9EQtebF7NaX9xuWi2R3YUgji06Gk4/RPIjs7wKy3pG+3/Zjr/1gsrZlwnSxNiLOA6jPtIPjjIPbg8a3FZ4LVrtEO2MLWtqJHQwhS8bxShISCcdfCoLtL2P6X13eGrvcHpsKYlvo3HIikJ6YD5u9vJOSOWezh1DGGhxBkZdHLmw3XqWEus5uoUqtFxs+tNIJmQ3PWLZc4ykKHI7qgUqSexQ4gjtFYf1rp67aH1jItUouMyYbwXHfTw305yhxJ7+B7jkcxWyNmGz6Ds/iy4dru90lxJKg4WJa0KS2scCpO6kYJGAfAV0Nc6K03rSAmJqC3IkdHnoXkkpdaJ60qHEeHI44ivlHXR0czg3Nh70liMrRfIqrNDekZp+Ta2WdWRpMG4IAS48w10jLn5WB7SSezB8a9Wp/SO0jBYxY4U67vnlvJ6BseJV7Xlu+dc64+jLY3HlKt+p7hHbPJLzCHSPMFP2V0NO+jhpGC+l673K43bdP4rIZbV47vte5QrM79Lvv59mf54ryPaLWXd2w69gxdij1/tEne+WWEx4SgcKy6DveCkpC/Ais1bF9C/w/wBYfJD0h2NDZjrfkPNgFSQMBIGeGSpQ8s1qLXOyPTWqbXa7UXpdpt1sC/V4tv6NDeV4yogpOTw5957a9ey3ZnYtngnm0yJklybudI5KUkqSE5wBupGB7R+FeaeuhpqZzYid8/g8F9fE57wXaKv/AO5m05/zju36jf7qqTbts1b2d3S3IhS5EyDOZUUuvJAUHEn2k8OGMKSfM1tOoptN0HZ9f2Vi2Xd2Swlh8PtuxykLScEEZUCMEHs6hWOkxaZsoMrrt4r7JTNLfdGag3onan+WNn7lkfc3pNme6MAniWV5Ug+R3x4AVYmv9VQNH6bfu80hSh7EdkHBecPJI+0nqANRHQ+zPTOyyTO1IxfrkmOIqkSRLcbLW7kEHCUA7wI4Y7cddUZtX1tJ1rqRUr227fHy3CYV9FPWoj6yuZ8h1Vs09AzEawuj/d6n0+aiMXxb9Mpf/wBhyA+vYFH9Q3iffrzJu9yeLsqSvfWeodgHYAMADsFeClKvLWhjQ1osAuUve6Rxc43JSlKV6XlKUpRFzdS3ZiyWSTcnyCGkeynPz1HglPmazhOkvTZj0uSsreeWVrUesk5NTjbHqL5SvAtEZzMWEo9IQeC3ev8AV5eOagNULH6/2ifo2/C3z4+i6jsvhnslN0rx7z8+wcPVKUpUCrOlKUoiUpSiJSlKIpZsy1KdP3wIkLxAlEIfzyQepfl19xNX0CCAQcg8jWWauPY/qr1+GLFOczKjp/i6lH8Y2OrxT9ngatWzuI7p9mkOR09FR9rMI32+2RDMfF2c/lx6uxWJSlKuK5+lKUoiUBIORwNKURaM2EbTxd22tM6hkf8AxFA3YklZ/wDmAPoKP1x2/S8edyVhBta2nEuNrUhaCFJUk4II5EGtI7GdrDN8SxYNRuoZuoAQxIUcJldgPYv4Hq48KpeNYMYyZ4B7vEcuvs8uxdE2d2hEwFNUn3uB59R6/Pt1t6lKVWFdEpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREr5S5DESK7KlPIZYaQVuOLOEpSBkknsr8z5cWBDdmTX248dlJW444rCUgdZNZl2zbUH9WOqs9oLjFkbXkk8FyiOSlDqT2J8zxwBv4fh8tbJut04nkovFcWhw2Lffm46Dn9uZXl2z7R39Y3EwLetbVkjL/BI5F9Q/nFD7B1eNV1SldFpqaOmjEcYsAuS1lZLWTGaU3J/LDqSlKVnWqlKUoiVFtpWpRp6xK6BY9ek5bjjrT2r8vtIqQ3KZGt0B6dLcDbDKCtaj1D99Z41ffZGor27cH8pR81lvP4tA5Dx6z3moXG8R9kh3WH33adXX6Kx7OYQa6o33j3G69Z4D16u1chRKlFSiSScknrr+UpXPl1VKUpREpSlESlKURKUpREr7wJciDNZmRXC0+ysLQodRFfClfQSDcL45ocCDotFaJ1HG1JZ0y291EhGEyGs/MV+48x/qru1nDSF/lacvDc6PlSD7LzWeDiOsePYa0HZrlEu1tZuEF0OMOjIPWD1g9hFdBwfExWR7r/AIxr19fquU7QYMcPm32D9m7Tq6vTqXspSlTKryUpSiJQEggg4I5GlKIr52NbXwrodP6uk4PBEa4OHn2JdP8Aa9/bV7AggEEEHiCKwhVlbLdrF10oWrdc+kuNmBwGycusD8gnq/JPDsxVUxXAN4mWmGfEenorzgm1G4BBWHLg719e/mtTUrm6dvlq1Da27lZ5jcqMv6STxSfqqHMHuNdKqg5pabEWKvrXNeA5puClKUr4vSUpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoiV4L/eLbYbW9c7tLbixWhlS1nmeoAcyT1AVG9pG0Wx6KilEhYl3JactQm1e0ewrP0U954nqBrMWt9YXzV9y9cu8kqSknoY6ODTIPUkfeeJ7amcNwaWsO87JnPn2eqr+MbQQYeCxvvScuXb6arubV9pFx1rMMZnfiWZpWWY2eKz1LcxzPdyHxMDpSr5T08dPGI4xYBcvqquWrlMsxuSlKUrMtdKUpREpSq42s6xEJhyw2x3+NODElxJ/FJP0R+UfgPHhrVlXHSRGV/8AnqW7h9DLXTiGIZnwHMqObWNWi7zPki3uZgx1/hFpPB5Y/sjq7Tx7KgVKVzWrqn1UplfqV2ChooqKBsMQyHj1pSlK11tpSlKIlKUoiUpSiJSlKIlKUoiVKdnurHtNXHddKnLe+R07Y+ifrp7x8R5VFqVmgnfBIJIzYhYKmmjqojFKLtK1DDksTIrcqK6l5l1IUhaTkKFfaqJ2c6zd07KESYpblsdV7SeZaP1k93aP9jeMWQzKjtyYzqHWXEhSFoOQoHrFdEw3Eo66O4ycNR+cFyXGMIlw2Xddm06Hn919aUpUiohKUpREpSlEXa0fqm96UuQn2WYplZ4ONn2m3R2KT1/aOoitEbO9sVh1H0cK7FFouSsABxf4F0/krPI9yvImsu0qMr8JgrRdws7mPzNTOF45VYcbMN28jp8uS3hSsobPdrGo9KBuI8r5UtieAjPrO82PyF8SnwOR3VfWidpulNVbjMab6nOVw9UlYQsnsSeSvI57hVLrcIqaQkkXbzH15LomHY/SVwAa7ddyP05/mSmtKUqLU2lKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlK+UqQxFjrkSXm2GWxvLccUEpSO0k8BVV6224aftPSRrA0bxLGR0gO4wk/nc1eQwe2tinpZql27E261auup6Nm/O8NH5oNSrPudwg2uC5OuMtmJGaGVuurCUjzNUdtE25qX0tv0c1up4pNweRxPe2g8vFXu66qfWOrr/AKsnetXqct4JJLbKfZaa/NTyHjzPWTXu0hs71bqhtL9sta0xVcpMg9G2e8E8Vfog1aqTA6elb0tY4Hq4fdUiu2lqq13QUDSOvVx9PzRRmXJkTJTkqW+4++6oqcccUVKUTzJJ5mvlVoz9hWto0YvMuWqYsDPRMyFBR7hvpSPjVb3SBNtc52DcYrsWS0d1bTqd1STVgp6unnyhcDbkqpV0NVTG87CL8T6rzUpStlaaUpSiJSlRLaDrGNpuIWGCh65Op/Bt8wgfWV3dg66w1FRHTxmSQ2AWxS0stVKIohclfLaTrFrT8MwoS0rubyfZHMMpP0j39g/2NGOuLddW66tS3FqKlKUckk8yTX0mSZEyU5KlOreedUVLWo5JNfGud4liL66XeOTRoPziutYPhMeGw7jc3HU8/slKUqOUslKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoiVLtn2s5OnJAjSd9+2OK9tvmWz9ZP3jrqI0rNT1ElPIJIzYha9VSxVcRilFwVp+3zItwhtzIT6H2HRlC0ngf9uyvRWedF6ruGmpm8yS9EWfw0dR4K7x2K7/AH1eenb5br9AEy3PhaeS0Hgts9ih1Vf8MxaKtbbR/Eei5ZjOBTYa/e1YdD9D1+a6dKUqWUElKUoiVYGzrZTftYQhcg8zbrcokIfeSVKdxwO4kcwDwySO7PGq/raug1Q16JsioG76r6gyG8HOBuDge/t781C43iEtHE0xak68lY9m8KhxCdwm0aL25/ZUVftgeoYkdTtpukO5KSM9EpJZWrwySn3kVVV3tlws9wcgXOG9ElNn2m3U4I7+8d44Vf8AqfaNrHSm0z5OvFtYXY5L4TG3GyCpokAKSvrUM8QevhwyDU72naOgaw02/EfZSJrSCuG+AN9CwOAz9U8iPvANR8OL1NMWe1Wc1+hH5w4qUqMApKxsnsN2vYbFp49+efDNZ30TtY1ZpoIYMr5Tgp4eryyVbo7Er+cnw4juq4dLbcNJXTdauiZFmfPPpR0jWe5aRn3gVmEgg4IwRX6S24ptTiUKKEEBSgOAzyyfI1KVeC0lT7xbunmMvsoah2irqQBodvN5HPx18VuW3T4VxiplW+ZHlsK5OMOBaT5jhXprDlnu90s8oSrVcJUJ767DpQT3HHMdxqxLFty1lACUTxBuiBzLzW4vHcUYHvBqvVGzU7DeJwcO4+nirXSbY0sgtO0tPePXwWn6VTtj2+6fk4Rd7TOgLP0mlJeQPE+yfganNp2h6JuiAqLqa3JJ5Jfd6FXuXg1DzYfVQ/Gwj85qfp8Voqn93KD87HuOalNK+MSVGltdLEksyG/rNLCh7xX2rTUglKUoiUpSiJSlKIlK8NxvFot3/CF1gw8f3+Qhv7TUUvm1nQlqCgq9omuDk3DQXc/pD2fjWaKnllNo2k9gWCaqggF5XhvaQFOaVRV99INobyLHp5avquzHce9Cc/1qgGotrmubylbRuot7KubcJHRY/S4r/aqVg2frJfiAaOs+ig6narD4fhcXHqH1NlpvUeqdPada371d4sM4yG1Ly4odyBlR8hVVaq2/QWQtnTVpclOchIlncQD2hA4keJTWf3XHHnVOuuKccUcqUo5JPeamekdl+sdSBD0e2mHEXxEmZltBHaBjeUO8AipmPAqOlbv1L79uQ9fFV6XabEK53R0cduwXPoO75rj6s1fqLVMjpb1c3pCAcoZB3WkeCBw8+ffXCrrawsMvTOpJlkmkKdir3d9IwFpIBSoeIINcmrFA2MRjogA3hZVKpdM6V3TklwyN9Vcfo+7Ool939TXxgPQWXNyLHWMpeWOalDrSOWOs5zywbg17r3Tuh2GW7itxchxGWYkZIKykcM4JASnhjJ8s4r4bD+h/3K7F0GN3oVZx9bpFb3xzVLek3b5cbaEic7vGPMiILKs8BueypPkeP6VU8D9TxJ0c590XsOz8uV0AuODYOyWmaC51iT28floFdmz7aLp7WqnWLct+PMaTvKjSEhKyn6wwSCOPbnurm7cdDsaq0y7OjMj5XgNlxhaR7TqBxU0e3PHHYfE1mTS95l6f1BCvMJRD0V0LwDjfT9JJ7iMg+NbZhvolRGZLWdx5tLic9hGRWLEaQ4VUMlgOR+mo7FmwivGOUkkFSMxr89D2hYUpXc2gxGYGur7Dj7oZanvJQEjgkb5wPLl5Vw6vEbw9gcOK5tLGY3lh4GyUpVcbQdobUJLlssLqXZXFLkkcUtdye1XfyHf1YKusipI+klP37Fs0OHz10oihFz4DtXV2ha3jafZVChKQ/c1DgnmlnP0ld/YP9jSEyTImSnJUp5bzzqt5a1HJJr5uuOOuqddWpxxZKlKUckk8yTX5rn2I4lLXPu7Jo0H5xXVcJweHDY91mbjqef2SlKVHKWSlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESvfY7vcLLPTNt0hTLqeBHNKx2KHWK8FK9Me5jg5psQvEkbZGljxcHgr60Trm26hQmM6UxLjjiyo8F96D1+HPx51LayylSkKCkqKVA5BBwQasjRe0x+IEQtQBchkcEykjLifzh9Id/Pxq4YbtC11o6nI8/VUHGNlHMvLR5j+nj8ufZr2q3qV57fNiXCIiVCkNyGFjKVoVkH/bsr0VaAQ4XCpLmlps4WKt/Z3sjhau2etXj5Teh3B55wNndC2yhJ3QFJ4HOQeOevka6ejdS3LZLqAaL1Y8xItboDzT7Civ1cLJGcYzukgkpxkcxnPH7+jXra3R7evSNyfRHfLxchLWcJc3sZbz9bPEduffJNs+y1/WU1m8WiYxHuDbQZcbfyEOpBJByASCMkcuPDliqlUVLva301af2Zvbq5EFXykpG+wx1mHD9q2wOevMEf48lOr/AGayausIizUNy4b6Q4y82oZScey42ocj2EfZVFX7WGv9mV9f0/OuAu8It70V2WkqUps5AUlfzgRjBBJAI86uLZPpq4aT0ZHs9zmolPocWv8ABklDYUc7iScEjmeQ4k1U/pTSGJuo7FaYaOmntNLK0NjeUekUkITgdfsnh3jtrTwrcNUaU2fHnr1cRyUhjhe2ibWtvHKLaHW/8J521VRaes9wv95jWm2MF6VIXupHUO1RPUAOJNa70Do22aS0yizsNofU4N6W6tAPTrI4kjs6gOoeZrhbFdnzWjrN63OQhd6loBfVz6FPMNpP2nrPcBXm1VrrpdqNi0XaXshMoLuLiDzISVBr4ZPkO0VmxOtkxGUwwfA25J5249nJa+DYdFhMIqKn948gAcr8O3n1LlbddBaViaHuWoLfaGYVwjlopVHyhB3nUJOUD2eSj1Vna3xH58+PBjI335DqWmk5xlSiAB7zWtduSN/ZTfU9jSD7nUH7qzxsQt/yjtRsjRTlLLxkKPZ0aSofECt/BKp4oJJHm+6Tr1AFRe0lDGcTiijaBvgaZZlxF18rrsx13bcl/Tcx1I6426/n9Qk1ECCDgjBreFZL27aa/g5r+UWW92HcP42xgcBvH20+Ss8OwismEY06skMUoANsrLFj2zrMPiE0LiW3sb8OWllBmXnWHA4y6tpY5KQog+8V3rfrjWMABMXU11QlPJKpKlpHkokVdGxrZ5pi87NIUy+2dmVIlOOuBwqUhYTvlIG8kg49nPnVZ7ctOWLS2r2rXYm3m2zFS66lx0r3VKUrgM8eQHvrajr6WrqHUxZci+oBGS0pcMraGkbVtks11tCQc/zmvgxtY2hM/M1G6fz47SvtQa9jW2baCj513Zc/OhtfckVXtXbpvYQzdbDb7qvVC2xNitSOjTBB3N9IVjO/xxnnilZHhtM0GZjRf/j6BfMPlxescW08jiRr73qVGl7bNeqTgTYaT2iInPxrzO7Y9oa/m3xDf5sNn70VYjfo920fjNSy1fmxkj7zXNv/AKP0hqGt2yX5Ml9PFLElncCu7fBOD4jHeK0Y6rBd6wA+bfUKTlotot25c75OF/AqASNqOv387+pZQz9RCEf1UiuHctTajuXC4X65yh9V2UtQ9xOK58mLJjzXIT7C25LbhaW0oe0lYOCMdua1NYdj+iIdviibZRKmJaQHnHJDhCl4G8d0Kxzz1VuVlRRYcGuMY97SwH2Ufh9LiWLFzRKfd13nO496ymSSck5Nei3wZtwkCPAhyJbx5NsNlaj5DjV3+kFs8tlu09Hv2nbazDREV0cttlOApCj7Kz3g8Cfyh2VE/RuvHybtGbhrXhq5MLYOeW+PbSfH2SP0qzMxJs1G6phF7Xy7PtmsEmDup69tJO628RmOv75Lz6f2N64uu6t6C1bGj9OY7un9VOVe8CrBsno/WttsKvV+mSF9aYraWkjuyreJ+FXHdVS0WyUu3pbXMSyssJcBKS5undBwQcZx11nzQ20XaZqTWkBtjMiGZKEymGoaQ0hskbxUrGU4GTkq9/KoCPEMQr2PfG9rA3Xh6/RWiXCsKwySOOVjnufpx8Bb6r07S9ice02OReNNTZTwioLr0WRhSigcSUKSBxA44I5dfbYmwXUX8INnsRDrm9Kt/wDFHsniQkDcP6pHHtBqQbQb1FsOkbhOkgrPQrbaaSkqLi1JISnA+PYM1n/0bNRfJOuFWl5zdjXVvouJ4B1OSg+ftJ8VCsTTPiOHvMmZYbg+Y7lmeKbCcWjbDkJBYjgM8j3qTelPp3Crdqhhvn/FJJA8VNn+sM+FURW1Ne2FvUukLlZVhO9IZPRE/RcHFB/WArHNus12uNxVboFulSpaVFKmWmipSSDg5A5Y76ltn6wSUpY4/B5cFBbV4eYq0SMGUnmNfoVdXox6wZS09o+c6EuFan4JV9LIytvx4bw8VVaW0bR1v1rYDbZqiy6hW/GkJTlTS/DrB5EdfiAao2xbFdYogO3d+W1bJkdtT0Vhte+8pxIykZScJJOOIJI7K6ektvc6JERF1JavXloSEiSwsIWrH1kngT3jHhUfW0ZnqDU0DrkHO3A8xwN1LYdiApqRtJijC1pBsSNRyPEW4L66d2ATk3ZC79d4ioCFZKIm8XHR2ZUAE/Grh1nqO16N005cphShtpG5HYTwLi8eyhI/2wONVPfvSCSY6kWKwKDpHsuzHRhP6Cef6wqnNU6kvOp7kbhepzkl7kgHghsdiUjgB4VlbhtdiEjXVps0cMvosD8Yw3ConMw8bzncc7eOtuQXhuUx+43GTcJSt+RJeW86rtUokk+81z7jNiW6IuXOkNx2GxlS1nA/1nuqPax1vadPJUxvetzscI7avmn8o/R+3uqmdT6jumopfT3B/KEn8Gyjg234Dt7zxqRxDGoKMbjPedy4Dt9FE4Vs7U4gekk91h4nU9nrp2qS682hSbuF2+0dJFgn2VucnHh/ZT3cz19lQKlKo9VVS1T9+U3P5ouk0VDBRRCKFth59qUpStdbaUpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoi6mnr/dbDJ6e2ylNZPttnihfin7+dW1pLaParruR7nu26WeGVK/BLPcrq8D7zVI0qSocVqKM2Ybt5HT7KIxLBKXEBeQWdzGv3+a1MlQICknIPEEGrD0ftf1fp5lEVyQ3dYiBhLcwFSkjuWDve/IrG+mNYXywKSiLJ6WMOcd72keXWnyq0dN7SLFc91qao22QeGHTlsnuX+/FWmHFKHEG7k4API/Q/wCFSKjBcTwl5lpiSObfqP8AIWnLp6QF5ehqbt1ihw3yMdK48p0DvCcD4586k+xDREx6UrXurC5Jukz8JFS9xUhJH4w95HBI6h4jGb21ocQlaFJUlQyFJOQRUo0xr7V2nd1FsvckMJ5MPHpWsdgSrIHlistThIEBjo7NJ11zHK+aw0eOk1LZcQJeG6DLI87ZArYMi4QI8tmG/NjNSXwS0yt1KVuAc90E5PMcqiMfZhpiJq+NqeA3KizWXlPKQl4rbdUoEEkKyc8c8CKy9rHUt11Xe13e7OpU+pIQlKAQhtI5JSMnA5nxJqe7BNXagGu7bZZV8kLtbwcC2ZDm8ng2opCSrin2gOWKiH4HUUtO6SOSxsd4c+rr7lPR7SUtdVNilhuN4bp4g3yPV8iru2xo39mF/T2RCfcQfuqnfRXt/T6vudyUnKYsLowexTixj4IVV1bU0b+zfUKey3vH3JJ+6oF6K1v6HSVzuRThUqYGx3pbQMfFaq1aSbo8KmHMgd9vot6up+lxunPJpPdf62VpXe7wrU7b25jm56/LTEZPV0ikqUB57uPEioJ6ROmvlzQq7gw3vS7UoyE4HEtcnB7sK/RqL+lRdnYzmnoUZ1TbrbjkvKTxSpO6EEee9VqaGvkfVmjYN23UKEpjdkN4yAseytJHZkHyxWtHC+jjhrW8SfP6i625aiPEJqjDn8ALfMfQ2X10Nb/krRlmtxTuqYhNIWPyt0b3xzWW9ttw+UtqN7dCspafEdPd0aQg/FJrXT7rbDDjzit1ttJUo9gAyaw3dJa59zlTnfnyXlvK8VKJP21K7NtMk8sx/Lm/0UJtg8RU0NO3S/kLfVeetnbMl7+zrTquy2Rx7m0j7qxjWxtka9/Znp9XZCQPdw+6tracfsGHr+i0diz/AOTIP+P1VT7a9daw0/tGet9mvD0eMlppTbIaQoZKRnmk5yau3SMu4T9L2ybdY/q85+Mhb7e7u7qyOPDq8OqudqLV2jtP3UM3q5RIk4thYC2iV7pzg5APYaiWptuOkoENw2cv3aVyQhLamm89pUoA48AahHxy1kMccUFiP4ra/Ow81ZI5YaCollnqgQT8N9PlcnuAUJ1PZWLp6TrUFpO8gyGJEhOOHsNJcUPMJHvq49qGoHdMaIn3qPul9gthtKuSipxIx7iaqb0c0zNRa+v2rbkrpX0tbpUeQW6rPDuCUEAdQNWltT0nI1nptuzMXFEBPrKHXXFNFeUpCvZAyOsg8+qs9e5jaqGCY+6wNB8ytbC2yPoaipp2+9I5xb5DuN12Y7tt1RpdDu6H7fc4uSk9aFp4g9h447jWSbzAnaB2h9AreL1smIeZXy6RAUFIV5jGfMVqXZxph7SGmW7G7dVXJDTilNOFno9xKuO7jePDOTz66gPpNaT+ULEzqiI1mTbx0cnA4qZJ4H9FR9yj2V9weqjgqnQA3jfkPp36Lzj9DLU0LKkttKwXP17tVbsOQ1LiMymFb7LzaXG1dqSMg+41BdbbUNL6LnLs7saW7MaSFFiMwEpTvDI9okDr6s1+9gl4+WNmVuC17zsLehud25839gopr3ZdZdZajj3i4zJbHRMBlbcfdBcwokEqIPbjlUfFDTw1To6q+6L6dWilZ56qoomTUVt51jnyOvcups31tbtb2h6dBjvxlMO9E607gkHGQQRzB+41V3pI6fjWSZadZWdtEOWZQQ8WxgKcA30Lx2+ycnr4VbNotml9BWBTMcxrXASrfccfexvqxzUpR4nh+6qC297QYOrZcW12VSnLbDUXFPKSU9M4RjIB44Azz55PdUhhMZfXb9O0iPO9+VtCovHZmx4Z0dW4GXK1ud9R6rRWk7yxqDTVvvUfARLYS4Ug/NVyUnyII8q/k+ZYNNRHZUt632phxanHFq3W+kWTkn8pRJ7yayrpvaVqnTulzYLRKajs9KpxLxbCnEb2MpSTwAzk8s5J41GLpcZ90lql3KbImSFc3H3CtXvNbTNmnmV28+zL5W1stGTbGNsLd2O77Z3yAPHr8loLVu3myw99jTkB25ujgH3stMjvA+cr3J8az3c5ap9ykzltNMqkPKdU20CEJKiSQkEnA41xL7frRZGekuU5pgkZSjOVq8EjiarXU+1OU+FMWGP6sg8PWHgCs+CeQ881JB2H4O0gH3j8yfTwUQ5uK4+4Ej3RplZo+ep8VZd9vlrskb1i5S22Rj2U5ytfgnmaqjVu0q5XHfjWdKrfFPDpM/hljx+j5ce+oPNlSZshUmXIdfeX85biionzNfGq/X4/PUXbH7rfHv8ARWrDNl6aks+X33deg+Xqv6olSipRJJOST11/KUqBVnSlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURdew6kvdkUPk6e603nJaJ3mz+ieHnVg2DauyvdavcAtnrejcU+aTxHkTVT0rfpcTqaXKN2XI5hRdbg1HW5ysz5jI9/qtJ2bUNlvCR8nXFh9R/m97dWP0Tg/CupWWUkpUFJJBHEEdVSK0a31NbN1LNzdebH83I/CDHZk8R5GrDT7TtOU7PmPQ+qqlXsY4Z00l+p3qPRajgaz1RCtUm0s3qUqBJZWw5HdV0iNxSSCAFZ3eB6sVKtnu126aQsjNlbtEGXEaUpQJUpDiipRJyrJHXjlWYbVtaUAE3W0g9rkZeP2VfvqUW3aJpabgKnLirP0ZDZT8RkfGpAVWF1jd0kZ52OWfgoo0WNYe8PaHZCwI97LxyVtbVNZnXGoWbr6kqEhqKlhLJd38YUpROcDrV2dVTb0cdcQLEufY73PaiQ3iJEd15W6hLnBKk56sjB/RPbVJQrhAmjMOdGkj/BOpV9hr1VuSUME1L7O34eFuCj4cSqaat9qd8fG+V1r3aNqS2DZpfp1uucSUPU1tpWw+lYBc9gHge1QrIVKV4w3Dm0DHNDr3N1kxjF34nI17m7thbW6Vr7Ymvf2WWFXYwoe5ah91ZBqS2nX2sLTbGbZbr7JjRGAQ20gJwkEknjjPMmvGL4e+uiaxhAIN8+wrJgGKR4bO6SQEgi2XaFMfSjRu7RIqvr2xs/0jg+6qproX++Xa/wAxEy8z3pshDYbS46ckJBJx4ZJ99c+tuigdTwMicbkBaGI1LaqqfM0WDjdaf9Gi1eo7OvXlJwu4Slug9e4n2APelR868W3DaZetH32JaLK3CUXYofdW82pak5UoADCgPonmDVAI1FqBEJuE3fLkiK0ndbYTKWG0jsCQcCuc86684XHnFuLPNSlEk+dRTcED6t1ROQ4G+Vu7uU4/aQx0LKWmaWkAe9fv7yrPse2nV41HBeu09pdtDyfWWG4yE5bPA8cb2QDkceYq371tT2cKhvw5d6bltPNqbcabjurC0kYIyE45d9ZPr8rUlCSpaglI5knAFZ6nA6WZwcBu2/psPotak2lrqdjmE79/6rn6qzNn20ePoFy+Q7dEcu0GS+Fwi450W6BvDeUMHiU7uR3V+dQbadb3TeRGkx7W0eG7Fa9rH5ysnPhiqduOqtOW8H1q8RARzShe+r3JyajNz2q2NjKYUSXMUORIDaD5nj8K8zNwyKQyylpd15nu+y9UzsZmiEMAcGDS2Qzz1y81ZVzuVxukgyLlPlTHj9N91S1e8mvC+80w0p191DTaeKlrUAB4k1S922oaglZTCRGgIPIoTvr96uHwqIXO6XK5u9JcJ0iUocukWSB4DkPKtSfaSnjG7C0nwH58lv02yFXMd6oeG+J9PFXXfNoum7cFIZkKnvD6McZT+seHuzVf3/aXf7hvNwujtrJ/vXtOY71H7gKhFKgarHKuoy3t0dXrqrRRbNUFLnu7x5uz8NF9H3nZDynn3VuuLOVLWoqUT3k186UqHJvmVPAACwSlKUX1KUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlESlKURKUpREpSlEX9BIIIJBHIiupC1HfoQAjXic2kck9Mop9x4UpXtkj2G7TZY5Io5BZ7Qe3NdeLtE1YxgG5JeSOpxlB+IGa60DahqJx1Lbke2rB6y0vPwVSlT+HVc7vieT8yqvi9BSs+GNo+QU2s2p581AU6zGBx9FKv3134s1135yUDwBpSrbA5xAuVQ6ljWuNgvU+6ptOQB51ypl3kspJSho4HWD++lKzSEgZLBC0E5hQ2/bQLzAz0MaArH121n7FVG5O07VDo/Brhx/wDs2M/1iaUqtYhUzMB3XkfMq4YVSU8jhvMB7QFypWttVSc9JepKc/3vDf8AVArjS5syYrely35Cu11wqPxpSqrJUSyfG4ntN1eIqWCH92wDsAC89KUrCs6UpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoiUpSiJSlKIlKUoi//9k=" style="height:48px;width:auto;object-fit:contain;mix-blend-mode:screen;" alt="Logo Red San Pablo">
    <div>
      <div class="topbar-title">RED DE SALUD SAN PABLO - 2026</div>
      <div class="topbar-sub">Seguimiento de Familias Sensibilizadas</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
  <a href="/" class="btn-home">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#0b1e42" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V9.5z"/><path d="M9 21V12h6v9"/></svg>
    ← Inicio
  </a>
  {f'<div class="fecha-badge"><span class="fecha-dot"></span>ACTUALIZADO &nbsp;<strong>' + (_fecha_badge or "") + '</strong></div>' if _fecha_badge else ''}
  </div>
</div>

<div class="content">
  <div class="hero">
    <h1>FAMILIAS SENSIBILIZADAS PARA LA PROMOCIÓN DE PRACTICAS Y ENTORNOS SALUDABLES</h1>
    <div style="display:inline-flex;align-items:center;gap:10px;margin-top:8px;flex-wrap:wrap;">
      <span style="display:inline-flex;align-items:center;gap:6px;background:#e8f0fe;border:1px solid #c7d7fc;border-radius:20px;padding:5px 14px;font-size:.78rem;font-weight:700;color:#1a3a7a;letter-spacing:.02em;">
        Panel de Seguimiento
      </span>
    </div>
  </div>

  <form method="get" action="/familias-sensibilizadas">
    <div class="filters">

      <!-- IPRESS -->
      <div class="filter-field">
        <label class="filter-label">IPRESS</label>
        <div class="ms-wrap" id="ms-ipress">
          <button type="button" class="ms-trigger" onclick="toggleMs('ms-ipress')">
            <svg class="filter-icon2" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
            <span class="ms-label" id="ms-ipress-label">{ipress_label}</span>
            <svg class="filter-arrow2" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
          <div class="ms-dropdown" id="ms-ipress-drop">
            <div class="ms-search-wrap"><input class="ms-search" type="text" placeholder="Buscar..." oninput="filterMs('ms-ipress', this.value)"></div>
            <div class="ms-select-all-row">
              <button type="button" class="ms-opt ms-all ms-all-btn" id="ms-ipress-all" onclick="selectAllMs('ms-ipress', this)">
                <span id="ms-ipress-all-icon" style="font-size:.85rem;line-height:1;">-</span>
                <span><b>Seleccionar todos</b></span>
              </button>
            </div>
            <div class="ms-list" id="ms-ipress-list">
              {ipress_opts}
            </div>
          </div>
        </div>
      </div>

      <div class="filter-divider"></div>

      <!-- MES -->
      <div class="filter-field">
        <label class="filter-label">Mes de Atención</label>
        <div class="ms-wrap" id="ms-mes">
          <button type="button" class="ms-trigger" onclick="toggleMs('ms-mes')">
            <svg class="filter-icon2" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
            <span class="ms-label" id="ms-mes-label">{mes_label}</span>
            <svg class="filter-arrow2" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
          <div class="ms-dropdown" id="ms-mes-drop">
            <div class="ms-select-all-row">
              <button type="button" class="ms-opt ms-all ms-all-btn" id="ms-mes-all" onclick="selectAllMs('ms-mes', this)">
                <span id="ms-mes-all-icon" style="font-size:.85rem;line-height:1;">-</span>
                <span><b>Seleccionar todos</b></span>
              </button>
            </div>
            <div class="ms-list" id="ms-mes-list">
              {mes_opts}
            </div>
          </div>
        </div>
      </div>

      <div class="filter-divider"></div>

      <!-- DNI -->
      <div class="filter-field">
        <label class="filter-label">Buscar por DNI</label>
        <div class="filter-icon-wrap">
          <svg class="filter-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <input type="text" name="dni" class="filter-input" value="{dni_raw}" placeholder="Ej: 70286548, 48066431">
        </div>
      </div>

      <div class="filter-divider"></div>

      <!-- ACTIONS -->
      <div class="filter-actions">
        <button type="submit" class="btn-filter">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
          Aplicar
        </button>
        <a href="/familias-sensibilizadas" class="btn-clear">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          Limpiar
        </a>
      </div>

    </div>
  </form>

  <!-- METRICS -->
  {metrics_html}

  <!-- TABLE -->
  <div style="display:flex;justify-content:flex-end;margin-bottom:0.5rem;">
    <button type="button" class="btn-excel" onclick="descargarExcel()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:5px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
      Descargar Excel
    </button>
  </div>
  <div class="table-wrap">
    <table>
      <thead>{ths}</thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  {'<div style="text-align:center;margin-top:2rem;"><span style="font-size:2rem;">🔍</span><p style="color:#f59e0b;font-weight:700;font-size:1rem;margin-top:.5rem;">' + aviso + '</p><p style="color:#94a3b8;font-size:.78rem;">Verifique el número e intente nuevamente.</p></div>' if aviso else ''}
  {'<p style="color:#64748b;font-size:.8rem;margin-top:1rem;text-align:center;">No se encontraron pacientes para los filtros seleccionados.</p>' if (df_final is None or df_final.empty) and not aviso and not error else ''}
  {'<p style="color:#dc2626;font-size:.8rem;margin-top:1rem;">Error técnico: ' + str(error) + '</p>' if error else ''}
</div>

<!-- POST-IT JS -->
<style>
.posit {{
    position: fixed; z-index: 99999;
    min-width: 220px; max-width: 290px;
    background: #fffde7; border-top: 38px solid #fdd835;
    border-radius: 4px 4px 6px 6px;
    box-shadow: 3px 6px 18px rgba(0,0,0,0.22), 1px 2px 4px rgba(0,0,0,0.10);
    font-family: 'Inter', sans-serif;
    padding: 10px 14px 14px 14px;
    animation: posit-pop .17s ease;
}}
@keyframes posit-pop {{
    from {{ transform: scale(0.88) translateY(-8px); opacity:0; }}
    to   {{ transform: scale(1)    translateY(0);    opacity:1; }}
}}
.posit-topbar {{
    position: absolute; top: -34px; left: 0; right: 0; height: 34px;
    display: flex; align-items: center; justify-content: space-between; padding: 0 10px;
}}
.posit-label {{ font-size: 11px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: #5d4037; }}
.posit-close {{ background: none; border: none; cursor: pointer; color: #5d4037; font-size: 17px; font-weight: 700; opacity: .75; padding: 0 2px; line-height: 1; }}
.posit-close:hover {{ opacity: 1; }}
.posit-pac {{ font-size: 12px; font-weight: 700; color: #5d4037; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px dashed #f9a825; }}
.posit-body {{ max-height: 260px; overflow-y: auto; }}
.posit-body::-webkit-scrollbar {{ width: 4px; }}
.posit-body::-webkit-scrollbar-thumb {{ background: #f9a825; border-radius: 2px; }}
</style>
<script>
var activePostit = null;
function removePostit() {{
    if (activePostit && activePostit.parentNode) activePostit.parentNode.removeChild(activePostit);
    activePostit = null;
}}
function togglePostit(btn) {{
    if (activePostit && activePostit._srcBtn === btn) {{ removePostit(); return; }}
    removePostit();
    var data    = btn.getAttribute('data-tip') || '';
    var pacname = btn.getAttribute('data-pac') || 'Paciente';
    var div = document.createElement('div');
    div.className = 'posit'; div._srcBtn = btn;
    div.innerHTML =
        '<div class="posit-topbar">' +
            '<span class="posit-label">📋 Pendientes</span>' +
            '<button class="posit-close" onclick="removePostit()">✕</button>' +
        '</div>' +
        '<div class="posit-pac">👤 ' + pacname + '</div>' +
        '<div class="posit-body">' + data + '</div>';
    document.body.appendChild(div);
    activePostit = div;
    var rect = btn.getBoundingClientRect();
    var scrollY = window.scrollY || document.documentElement.scrollTop;
    var scrollX = window.scrollX || document.documentElement.scrollLeft;
    var ph = div.offsetHeight || 200; var pw = div.offsetWidth || 240;
    var ww = document.documentElement.clientWidth; var wh = window.innerHeight;
    var top  = (wh - rect.bottom - 8 >= ph || wh - rect.bottom >= rect.top) ? rect.bottom + scrollY + 8 : rect.top + scrollY - ph - 8;
    var left = (rect.left + pw > ww) ? ww - pw - 12 + scrollX : rect.left + scrollX;
    div.style.top = top + 'px'; div.style.left = left + 'px';
    _skipNext = true;
}}
var _skipNext = false;
document.addEventListener('click', function(e) {{
    if (_skipNext) {{ _skipNext = false; return; }}
    if (!activePostit) return;
    if (activePostit.contains(e.target)) return;
    removePostit();
}});
</script>
<script>
/* ── STICKY THEAD: ajusta top de fila-2 según altura real de fila-1 ── */
(function() {{
  function fixStickyHeader() {{
    var thead = document.querySelector('table thead');
    if (!thead) return;
    var row1 = thead.querySelector('tr:first-child');
    if (!row1) return;
    var h = row1.getBoundingClientRect().height;
    thead.querySelectorAll('tr:last-child th').forEach(function(th) {{
      th.style.top = h + 'px';
    }});
  }}
  window.addEventListener('load', fixStickyHeader);
  window.addEventListener('resize', fixStickyHeader);
}})();
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
<script>
function descargarExcel() {{
    var tabla = document.querySelector('table');
    if (!tabla) {{ alert('No hay datos para descargar.'); return; }}
    var headers = [];
    tabla.querySelectorAll('thead th').forEach(function(th) {{ headers.push(th.innerText.trim()); }});
    var rows = [];
    tabla.querySelectorAll('tbody tr').forEach(function(tr) {{
        var fila = {{}};
        tr.querySelectorAll('td').forEach(function(td, i) {{ fila[headers[i] || i] = td.innerText.trim(); }});
        rows.push(fila);
    }});
    if (rows.length === 0) {{ alert('No hay datos para descargar.'); return; }}
    var wb = XLSX.utils.book_new();
    var ws = XLSX.utils.json_to_sheet(rows, {{ header: headers }});
    var colWidths = headers.map(function(h) {{ return {{ wch: Math.max(h.length, 12) }}; }});
    ws['!cols'] = colWidths;
    XLSX.utils.book_append_sheet(wb, ws, 'FamiliasSensibilizadas');
    var fecha = new Date();
    var nombreArchivo = 'FamiliasSensibilizadas_' + fecha.getFullYear() + ('0'+(fecha.getMonth()+1)).slice(-2) + ('0'+fecha.getDate()).slice(-2) + '.xlsx';
    XLSX.writeFile(wb, nombreArchivo);
}}
</script>
<script>
/* ---- MULTI-SELECT ---- */
function toggleMs(id) {{
    var wrap = document.getElementById(id);
    var isOpen = wrap.classList.contains('open');
    document.querySelectorAll('.ms-wrap.open').forEach(function(w) {{ w.classList.remove('open'); }});
    if (!isOpen) {{ wrap.classList.add('open'); var s = wrap.querySelector('.ms-search'); if (s) s.focus(); }}
}}
function filterMs(id, q) {{
    var list = document.getElementById(id + '-list');
    q = q.toLowerCase();
    list.querySelectorAll('label.ms-opt').forEach(function(opt) {{
        var spanEl = opt.querySelector('span');
        if (!spanEl) return;
        opt.style.display = spanEl.textContent.toLowerCase().includes(q) ? '' : 'none';
    }});
}}
function updateLabel(wrap) {{
    var labelEl = wrap.querySelector('.ms-label');
    var checked = wrap.querySelectorAll('.ms-list input[type=checkbox]:checked');
    var defaultLabel = wrap.id === 'ms-ipress' ? 'Todas las IPRESS' : 'Todos los meses';
    var unitPlur = wrap.id === 'ms-ipress' ? 'IPRESS seleccionadas' : 'meses seleccionados';
    if (checked.length === 0) labelEl.textContent = defaultLabel;
    else if (checked.length === 1) labelEl.textContent = checked[0].value;
    else labelEl.textContent = checked.length + ' ' + unitPlur;
}}
function updateAllChk(wrap) {{
    var allBtn = wrap.querySelector('.ms-all-btn');
    if (!allBtn) return;
    var iconEl = allBtn.querySelector('span:first-child');
    if (iconEl) iconEl.textContent = '-';
}}
function selectAllMs(id, btn) {{
    var wrap = document.getElementById(id);
    var allChecked = wrap.querySelectorAll('.ms-list input[type=checkbox]:checked').length;
    var total = wrap.querySelectorAll('.ms-list input[type=checkbox]').length;
    var nowChecked = (allChecked < total);
    wrap.querySelectorAll('.ms-list input[type=checkbox]').forEach(function(cb) {{ cb.checked = nowChecked; }});
    updateLabel(wrap);
}}
document.addEventListener('change', function(e) {{
    var cb = e.target;
    if (!cb.matches('.ms-list input[type=checkbox]')) return;
    var wrap = cb.closest('.ms-wrap');
    updateAllChk(wrap);
    updateLabel(wrap);
}});
document.addEventListener('click', function(e) {{
    if (!e.target.closest('.ms-wrap')) document.querySelectorAll('.ms-wrap.open').forEach(function(w) {{ w.classList.remove('open'); }});
}});
document.addEventListener('DOMContentLoaded', function() {{
    document.querySelectorAll('.ms-wrap').forEach(function(wrap) {{ updateAllChk(wrap); updateLabel(wrap); }});
}});
</script>
<script>
(function() {{
  var t0 = Date.now();
  function ocultarLoading() {{
    var ov = document.getElementById('rsp-loading');
    if (!ov) return;
    var elapsed = Date.now() - t0;
    var delay = Math.max(0, 900 - elapsed);
    setTimeout(function() {{
      ov.style.opacity = '0';
      setTimeout(function() {{ ov.style.display = 'none'; }}, 400);
    }}, delay);
  }}
  window.addEventListener('load', ocultarLoading);
  setTimeout(ocultarLoading, 2500);
}})();
</script>
</body>
</html>"""
    return html


IPRESS_DEFAULT = "SAN LUIS BAJO - GRANDE"

@familias_sensibilizadas_bp.route("/familias-sensibilizadas")
def familias_sensibilizadas():
    if not request.args:
        from flask import redirect, url_for
        return redirect(url_for("familias_sensibilizadas.familias_sensibilizadas", ipress=IPRESS_DEFAULT))

    ipress_sel = [v.strip() for v in request.args.getlist("ipress") if v.strip()]
    mes_sel    = [v.strip() for v in request.args.getlist("mes")    if v.strip()]
    dni_raw    = request.args.get("dni", "").strip()

    try:
        result = procesar_datos(ipress_sel, mes_sel, dni_raw)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(render_page(None, [], [], [], [], [], "", str(e)), mimetype="text/html; charset=utf-8")

    df_final, segundo, lista_ipress, lista_meses = result

    if isinstance(segundo, str):
        return Response(render_page(None, [], lista_ipress, lista_meses, ipress_sel, mes_sel, dni_raw, error=segundo), mimetype="text/html; charset=utf-8")

    if df_final is None:
        lista_dni = re.findall(r'\d+', dni_raw) if dni_raw else []
        if lista_dni:
            aviso = f"El DNI {', '.join(lista_dni)} no existe en el padrón." if len(lista_dni) == 1 else f"Los DNIs {', '.join(lista_dni)} no existen en el padrón."
        else:
            aviso = None
        return Response(render_page(None, [], lista_ipress, lista_meses, ipress_sel, mes_sel, dni_raw, aviso=aviso), mimetype="text/html; charset=utf-8")

    columnas_indicadores = segundo
    html = render_page(df_final, columnas_indicadores, lista_ipress, lista_meses,
                       ipress_sel, mes_sel, dni_raw)
    return Response(html, mimetype="text/html; charset=utf-8")