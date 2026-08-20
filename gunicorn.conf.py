# gunicorn.conf.py — Configuración para Render (plan gratuito, 512 MB)
import os

# Worker class "gthread" = 1 proceso + múltiples threads.
# Así el background thread del cálculo de avance (47 s) no bloquea
# las peticiones HTTP entrantes — cada request corre en su propio thread.
# NO usar gevent/eventlet: requieren instalar dependencias extra.
worker_class = "gthread"

# 1 solo worker = 1 sola copia del DataFrame en RAM
workers = 1

# 4 threads: 1 para peticiones HTTP mientras el background thread
# (descarga + cálculo) ocupa otro.
threads = 4

# Timeout elevado: el worker gthread responde peticiones aunque el
# background thread esté ocupado, así que 120 s es más que suficiente.
timeout = 120

# Keepalive: Render usa HTTP/1.1 con keep-alive
keepalive = 5

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

accesslog = "-"
errorlog  = "-"
loglevel  = "info"

# Preload: importa la app 1 sola vez antes de crear workers
preload_app = True
