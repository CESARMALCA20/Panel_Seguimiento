"""
parquet_cache.py — DataFrame compartido para todos los módulos Flask.
El parquet se lee UNA sola vez y se guarda aquí.
Cada módulo importa get_df() en lugar de leer el parquet directamente.

OPTIMIZADO PARA RENDER (plan gratuito ≈ 512 MB):
- Una sola copia del DataFrame en RAM (sin cachés secundarios en cada módulo)
- get_fecha_max(anio) expone la fecha máxima sin releer el parquet
- Los módulos NO deben mantener su propio _DF_CACHE
"""
import os, threading, time
import polars as pl

_dir = os.path.dirname(os.path.abspath(__file__))

_CACHE: dict = {}        # {anio: pl.DataFrame}
_FECHA_MAX: dict = {}    # {anio: str}  — para /api/update-time sin releer el parquet
_LOCK = threading.Lock()

MESES_ES = {
    "January":"Enero","February":"Febrero","March":"Marzo","April":"Abril",
    "May":"Mayo","June":"Junio","July":"Julio","August":"Agosto",
    "September":"Septiembre","October":"Octubre","November":"Noviembre","December":"Diciembre"
}

COLS_NECESARIAS = {
    'Fecha_Atencion','Nombre_Establecimiento','Numero_Documento_Paciente',
    'Apellido_Paterno_Paciente','Apellido_Materno_Paciente','Nombres_Paciente',
    'Fecha_Nacimiento_Paciente','Anio_Actual_Paciente','Genero',
    'Codigo_Item','Valor_Lab','Codigo_Diagnostico','Fecha_Ultima_Regla',
    'Lote','Apellido_Paterno_Personal','Apellido_Materno_Personal','Nombres_Personal',
    'Num_Pag','Num_Reg',
    'Descripcion_Item','Descripcion_Financiador','Id_Condicion_Servicio','Tipo_Diagnostico',
}

def _cast_fecha(col_name, df):
    if col_name not in df.columns:
        return df
    dtype = df[col_name].dtype
    if dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
        return df.with_columns(pl.col(col_name).cast(pl.Date))
    if dtype == pl.Datetime:
        return df.with_columns(pl.col(col_name).cast(pl.Date))
    if dtype == pl.Utf8:
        return df.with_columns(
            pl.col(col_name).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
        )
    return df

def _cargar(path: str) -> pl.DataFrame:
    """Lee el parquet y aplica transformaciones comunes."""
    todas = pl.read_parquet(path, n_rows=1).columns
    cols_leer = [c for c in todas if c.strip() in COLS_NECESARIAS]
    df = pl.read_parquet(path, columns=cols_leer if cols_leer else None)
    df = df.rename({c: c.strip() for c in df.columns})
    df = _cast_fecha("Fecha_Atencion", df)
    df = _cast_fecha("Fecha_Nacimiento_Paciente", df)
    df = _cast_fecha("Fecha_Ultima_Regla", df)
    cast_cols = [pl.col("Fecha_Atencion").cast(pl.Date),
                 pl.col("Fecha_Nacimiento_Paciente").cast(pl.Date)]
    if "Fecha_Ultima_Regla" in df.columns:
        cast_cols.append(pl.col("Fecha_Ultima_Regla").cast(pl.Date))
    df = df.with_columns(cast_cols + [
        pl.col("Fecha_Atencion").dt.month().alias("Mes_Num"),
        pl.col("Fecha_Atencion").dt.strftime("%B").alias("Mes_Nombre"),
        pl.col("Nombre_Establecimiento").str.strip_chars(),
        pl.col("Codigo_Item").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        pl.col("Valor_Lab").cast(pl.Utf8).str.strip_chars().fill_null(""),
    ]).with_columns(pl.col("Mes_Nombre").replace(MESES_ES))
    return df

def _parquet_path(anio: int) -> str:
    nombre = "reporte.parquet" if anio == 2026 else f"reporte_{anio}.parquet"
    return os.path.join(_dir, "data", nombre)

def _extraer_fecha_max(df: pl.DataFrame) -> str | None:
    """Extrae la fecha máxima de Fecha_Atencion del DataFrame ya cargado."""
    meses = {1:"ENE",2:"FEB",3:"MAR",4:"ABR",5:"MAY",6:"JUN",
             7:"JUL",8:"AGO",9:"SET",10:"OCT",11:"NOV",12:"DIC"}
    try:
        _max = df["Fecha_Atencion"].drop_nulls().max()
        if _max is not None:
            return f"{_max.day:02d} {meses[_max.month]} {_max.year}"
    except Exception:
        pass
    return None

def get_df(anio: int = 2026) -> tuple:
    """Devuelve (df, error). UN SOLO año en RAM a la vez.
    Libera el año anterior ANTES de cargar el nuevo."""
    import gc
    with _LOCK:
        if anio in _CACHE:
            return _CACHE[anio], None
        path = _parquet_path(anio)
        if not os.path.exists(path):
            return None, (f"El reporte {anio} no esta disponible aun. "
                          f"Espera unos segundos y vuelve a intentarlo.")
        try:
            t0 = time.time()
            # Liberar ANTES de cargar — nunca dos parquets en RAM
            _CACHE.clear()
            gc.collect()
            df = _cargar(path)
            _CACHE[anio] = df
            _FECHA_MAX[anio] = _extraer_fecha_max(df)
            print(f"[parquet_cache:{anio}] cargado en {time.time()-t0:.1f}s "
                  f"— {len(df):,} filas, {df.estimated_size('mb'):.1f}MB")
            return df, None
        except Exception as e:
            return None, str(e)

def get_fecha_max(anio: int = 2026) -> str | None:
    """Devuelve la fecha máxima cacheada sin releer el parquet.
    Si el año aún no se ha cargado, lee SOLO la columna Fecha_Atencion."""
    with _LOCK:
        if anio in _FECHA_MAX:
            return _FECHA_MAX[anio]
    # Intento rápido: leer solo 1 columna (muy liviano en RAM)
    path = _parquet_path(anio)
    if not os.path.exists(path):
        return None
    meses = {1:"ENE",2:"FEB",3:"MAR",4:"ABR",5:"MAY",6:"JUN",
             7:"JUL",8:"AGO",9:"SET",10:"OCT",11:"NOV",12:"DIC"}
    try:
        _df = pl.read_parquet(path, columns=["Fecha_Atencion"])
        _col = _df["Fecha_Atencion"]
        dtype = _col.dtype
        if dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Datetime):
            _col = _col.cast(pl.Date)
        _max = _col.drop_nulls().max()
        if _max is not None:
            resultado = f"{_max.day:02d} {meses[_max.month]} {_max.year}"
            with _LOCK:
                _FECHA_MAX[anio] = resultado
            return resultado
    except Exception:
        pass
    return None

def invalidar():
    """Limpia todo el caché (llamar desde api_refresh)."""
    with _LOCK:
        _CACHE.clear()
        _FECHA_MAX.clear()
    print("[parquet_cache] caché invalidado")
