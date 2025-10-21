import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import altair as alt

st.set_page_config(page_title="Regresión: mpg vs hp", page_icon="🚗", layout="centered")

# -------------------- ESTADO GLOBAL --------------------
def make_demo_df(n=160, beta0=50.0, beta1=-0.05, noise=4.0, seed=42):
    rng = np.random.default_rng(seed)
    hp = np.linspace(50, 300, n)
    mpg = beta0 + beta1 * hp + rng.normal(0, noise, size=n)
    return pd.DataFrame({"hp": hp, "mpg": mpg})

if "train_df" not in st.session_state:
    st.session_state.train_df = make_demo_df()
if "source" not in st.session_state:
    st.session_state.source = "DEMO"

st.title("🚗 Regresión lineal simple: mpg en función de hp")
st.caption("Sube un CSV con columnas **hp** y **mpg** para reemplazar los datos. Todo se actualiza automáticamente.")
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

if up is not None:
    try:
        df_new = pd.read_csv(up)
        missing = [c for c in ["hp", "mpg"] if c not in df_new.columns]
        if missing:
            st.error(f"Faltan columnas requeridas: {missing}. El CSV debe tener 'hp' y 'mpg'.")
            st.stop()
        df_new = df_new[["hp", "mpg"]].copy()
        df_new["hp"] = pd.to_numeric(df_new["hp"], errors="coerce")
        df_new["mpg"] = pd.to_numeric(df_new["mpg"], errors="coerce")
        before = len(df_new)
        df_new = df_new.dropna()
        if len(df_new) < 10:
            st.error("Muy pocos datos numéricos tras limpiar. Revisa tu CSV.")
            st.stop()
        if len(df_new) < before:
            st.info(f"Se descartaron {before - len(df_new)} filas no numéricas o vacías.")
        st.session_state.train_df = df_new.reset_index(drop=True)
        st.session_state.source = "CSV"
        st.success("Datos reemplazados por tu CSV. La app ya usa tu tabla para todo.")
    except Exception as e:
        st.error(f"Error al leer CSV: {e}")
        st.stop()

df = st.session_state.train_df

# -------------------- PREVIA --------------------
st.subheader("Vista previa de los datos actuales")
st.dataframe(df.head(20), use_container_width=True)

# -------------------- ENTRENAMIENTO --------------------
st.subheader("Entrenamiento del modelo")
test_size = st.slider("Proporción de test", 0.1, 0.5, 0.25, 0.05)
use_norm = st.checkbox("Normalizar X e Y (z-score)", value=False,
                       help="Se entrena en escala estandarizada, pero métricas y predicciones se reportan en la escala original.")

# Matrices en escala original
X_raw = df[["hp"]].to_numpy(dtype=float)
Y_raw = df["mpg"].to_numpy(dtype=float)

# Split
X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
    X_raw, Y_raw, test_size=test_size, random_state=42
)

if use_norm:
    # z-score con medias y desvs del entrenamiento
    x_mu, x_sd = X_train_raw.mean(), X_train_raw.std(ddof=0)
    y_mu, y_sd = Y_train_raw.mean(), Y_train_raw.std(ddof=0)

    # Evitar sd=0
    x_sd = x_sd if x_sd > 0 else 1.0
    y_sd = y_sd if y_sd > 0 else 1.0

    X_train = (X_train_raw - x_mu) / x_sd
    Y_train = (Y_train_raw - y_mu) / y_sd
    X_test = (X_test_raw - x_mu) / x_sd

    model = LinearRegression().fit(X_train, Y_train)

    # Predicciones en espacio normalizado y luego desnormalizar Y
    y_pred_norm = model.predict(X_test)
    y_pred = y_pred_norm * y_sd + y_mu

    # Coeficientes en escala original para mostrar ecuación: y = b0 + b1*x
    # Si y = (a0 + a1*zx)*y_sd + y_mu y zx = (x - x_mu)/x_sd
    # => y = (y_mu + y_sd*a0 - y_sd*a1*x_mu/x_sd) + (y_sd*a1/x_sd) * x
    a1 = float(model.coef_[0])
    a0 = float(model.intercept_)
    b1 = float((y_sd * a1) / x_sd)
    b0 = float(y_mu + y_sd * a0 - b1 * x_mu)

else:
    # Sin normalizar
    model = LinearRegression().fit(X_train_raw, Y_train_raw)
    y_pred = model.predict(X_test_raw)
    b1 = float(model.coef_[0])
    b0 = float(model.intercept_)

# -------------------- MÉTRICAS EN ESCALA ORIGINAL --------------------
mse = mean_squared_error(Y_test_raw, y_pred)
rmse = float(np.sqrt(mse))
r2 = float(r2_score(Y_test_raw, y_pred))

st.subheader("Ecuación del modelo (escala original)")
st.latex(r"\text{mpg} = \beta_0 + \beta_1 \cdot \text{hp}")
st.write(f"**β₀:** {b0:,.4f} | **β₁:** {b1:,.4f}")

m1, m2 = st.columns(2)
m1.metric("R² (test)", f"{r2:.4f}")
m2.metric("RMSE (test)", f"{rmse:.4f}")

# -------------------- GRÁFICA --------------------
grid = pd.DataFrame({"hp": np.linspace(X_raw.min(), X_raw.max(), 100)})
if use_norm:
    z = (grid[["hp"]].to_numpy(dtype=float) - X_train_raw.mean()) / (X_train_raw.std(ddof=0) if X_train_raw.std(ddof=0) > 0 else 1.0)
    grid["mpg"] = (model.predict(z) * (Y_train_raw.std(ddof=0) if Y_train_raw.std(ddof=0) > 0 else 1.0)) + Y_train_raw.mean()
else:
    grid["mpg"] = model.predict(grid[["hp"]])

scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X("hp", title="hp"),
    y=alt.Y("mpg", title="mpg"),
    tooltip=["hp", "mpg"]
)
line = alt.Chart(grid).mark_line().encode(x="hp", y="mpg")

st.subheader("Ajuste del modelo con los datos actuales")
st.altair_chart(scatter + line, use_container_width=True)

# -------------------- PREDICCIÓN SENCILLA --------------------
st.header("Predicción con dato nuevo")
x_new = st.number_input("hp (valor único)", value=float(np.median(X_raw)))
if use_norm:
    xz = (np.array([[x_new]]) - X_train_raw.mean()) / (X_train_raw.std(ddof=0) if X_train_raw.std(ddof=0) > 0 else 1.0)
    y_new = model.predict(xz).item()
    y_new = y_new * (Y_train_raw.std(ddof=0) if Y_train_raw.std(ddof=0) > 0 else 1.0) + Y_train_raw.mean()
else:
    y_new = model.predict(np.array([[x_new]])).item()
st.success(f"Predicción de mpg para hp={x_new:,.2f}: **{y_new:,.4f}**")