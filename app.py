import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import altair as alt

st.set_page_config(page_title="Regresión: mpg vs hp", page_icon="🚗", layout="wide")

# -------------------- DEMO DATA --------------------
def make_demo_df(n=60, beta0=50.0, beta1=-0.08, noise=3.5, seed=42):
    rng = np.random.default_rng(seed)
    hp = np.linspace(60, 260, n)
    mpg = beta0 + beta1 * hp + rng.normal(0, noise, size=n)
    return pd.DataFrame({"hp": hp, "mpg": mpg})

if "train_df" not in st.session_state:
    st.session_state.train_df = make_demo_df()
if "source" not in st.session_state:
    st.session_state.source = "DEMO"

st.title("🚗 Regresión lineal simple: mpg en función de hp")
st.caption("Sube un CSV con columnas **hp** y **mpg**. La ecuación se calcula por OLS sobre todo el dataset (igual que Excel).")
st.markdown(f"**Fuente de datos actual:** `{st.session_state.source}`")

# -------------------- CONTROLES DE ORIGEN --------------------
colA, colB = st.columns([3, 1])
with colA:
    up = st.file_uploader("Reemplazar datos con CSV (columnas: hp, mpg)", type=["csv"])
with colB:
    if st.button("Restaurar DEMO"):
        st.session_state.train_df = make_demo_df()
        st.session_state.source = "DEMO"
        st.success("Datos restaurados a DEMO.")

def _clean_and_validate(df_raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["hp", "mpg"] if c not in df_raw.columns]
    if missing:
        st.error(f"Faltan columnas requeridas: {missing}. El CSV debe tener 'hp' y 'mpg'.")
        st.stop()

    df = df_raw[["hp", "mpg"]].copy()
    df["hp"] = pd.to_numeric(df["hp"], errors="coerce")
    df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce")
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        st.info(f"Se descartaron {dropped} filas no numéricas o vacías.")

    # Validación de tamaño
    n = len(df)
    if n < 10:
        st.error("Muy pocos datos numéricos tras limpiar. Se requieren al menos 10 filas.")
        st.stop()
    if n > 100:
        st.error(f"Demasiados datos ({n}). Máximo permitido: 100 filas tras limpieza.")
        st.stop()

    return df

if up is not None:
    try:
        df_new = pd.read_csv(up)
        df_new = _clean_and_validate(df_new)
        st.session_state.train_df = df_new
        st.session_state.source = "CSV"
        st.success("Datos reemplazados por tu CSV. La app ya usa tu tabla para todo.")
    except Exception as e:
        st.error(f"Error al leer CSV: {e}")
        st.stop()

df = st.session_state.train_df

# -------------------- PREVIA --------------------
st.subheader("Vista previa de los datos")
st.dataframe(df, use_container_width=True)

# -------------------- CHEQUEOS BÁSICOS --------------------
X_raw = df[["hp"]].to_numpy(dtype=float)
Y_raw = df["mpg"].to_numpy(dtype=float)

# Variancia de X: si es cero, no hay regresión válida
x_var = float(np.var(X_raw, ddof=0))
if x_var == 0.0:
    st.error("La variable independiente 'hp' es constante. No se puede ajustar una recta. Cambia tus datos.")
    st.stop()

# -------------------- AJUSTE OLS COMPLETO (COINCIDE CON EXCEL) --------------------
ols_full = LinearRegression(fit_intercept=True)
ols_full.fit(X_raw, Y_raw)
b1_full = float(ols_full.coef_[0])
b0_full = float(ols_full.intercept_)

st.subheader("Ecuación del modelo (OLS en escala original, sobre todo el dataset)")
st.latex(r"\text{mpg} = \beta_0 + \beta_1 \cdot \text{hp}")
st.write(f"**β₀ (intercepto):** {b0_full:,.6f} | **β₁ (pendiente):** {b1_full:,.6f}")

# -------------------- MÉTRICAS (OPCIONALES) --------------------
st.subheader("Métricas de generalización (holdout)")
col_m = st.columns([1,1,1,1])

with col_m[0]:
    test_size = st.slider("Proporción de test", 0.1, 0.5, 0.25, 0.05)

with col_m[1]:
    use_norm = st.checkbox("Normalizar X e Y (z-score) para MÉTRICAS", value=False,
                           help="Esto solo afecta el cálculo de métricas. La ecuación y la recta mostradas SIEMPRE se basan en OLS con todos los datos en escala original.")

# Split para métricas
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, Y_raw, test_size=test_size, random_state=42)

if use_norm:
    x_mu, x_sd = X_tr.mean(), X_tr.std(ddof=0)
    y_mu, y_sd = y_tr.mean(), y_tr.std(ddof=0)
    x_sd = x_sd if x_sd > 0 else 1.0
    y_sd = y_sd if y_sd > 0 else 1.0

    X_tr_n = (X_tr - x_mu) / x_sd
    y_tr_n = (y_tr - y_mu) / y_sd
    X_te_n = (X_te - x_mu) / x_sd

    model_m = LinearRegression().fit(X_tr_n, y_tr_n)
    y_pred_te = model_m.predict(X_te_n) * y_sd + y_mu
else:
    model_m = LinearRegression().fit(X_tr, y_tr)
    y_pred_te = model_m.predict(X_te)

mse = mean_squared_error(y_te, y_pred_te)
rmse = float(np.sqrt(mse))
r2 = float(r2_score(y_te, y_pred_te))

with col_m[2]:
    st.metric("R² (test)", f"{r2:.4f}")
with col_m[3]:
    st.metric("RMSE (test)", f"{rmse:.4f}")

# -------------------- GRÁFICA: DATOS + RECTA OLS COMPLETA --------------------
st.subheader("Ajuste del modelo (OLS completa)")

grid = pd.DataFrame({"hp": np.linspace(float(df["hp"].min()), float(df["hp"].max()), 100)})
grid["mpg"] = ols_full.predict(grid[["hp"]])

scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X("hp", title="hp"),
    y=alt.Y("mpg", title="mpg"),
    tooltip=[alt.Tooltip("hp", format=".3f"), alt.Tooltip("mpg", format=".3f")]
)
line = alt.Chart(grid).mark_line().encode(x="hp", y="mpg")

st.altair_chart(scatter + line, use_container_width=True)

# -------------------- PREDICCIÓN SENCILLA (con OLS completa) --------------------
st.header("Predicción con dato nuevo (usando OLS completa)")
x_new = st.number_input("hp (valor único)", value=float(np.median(X_raw)))
y_new = float(ols_full.predict(np.array([[x_new]])).item())
st.success(f"Predicción de mpg para hp={x_new:,.3f}: **{y_new:,.6f}**")