import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import altair as alt

st.set_page_config(page_title="Regresión lineal simple", page_icon="📈")

st.title("📈 Regresión lineal simple")
st.caption("Elige DEMO para jugar con datos sintéticos o sube un CSV. Selecciona X (independiente) e Y (dependiente).")

# --------- MODO DE USO
modo = st.radio("Modo de datos", ["DEMO (recomendado)", "CSV"], horizontal=True)

if modo.startswith("DEMO"):
    # Datos sintéticos súper simples
    st.sidebar.header("Datos DEMO")
    n = st.sidebar.slider("Número de puntos", 30, 500, 120, 10)
    beta0_true = st.sidebar.slider("β₀ (intersección real)", -50.0, 50.0, 5.0, 0.5)
    beta1_true = st.sidebar.slider("β₁ (pendiente real)", -5.0, 5.0, 1.7, 0.1)
    ruido = st.sidebar.slider("Ruido (desv. estándar)", 0.0, 30.0, 10.0, 0.5)
    rng = np.random.default_rng(42)
    x = np.linspace(0, 100, n)
    y = beta0_true + beta1_true * x + rng.normal(0, ruido, size=n)
    df = pd.DataFrame({"x": x, "y": y})
else:
    st.sidebar.header("Datos CSV")
    file = st.sidebar.file_uploader("CSV con encabezados", type=["csv"])
    sep = st.sidebar.selectbox("Separador", [",", ";", "\t"], index=0)
    if not file:
        st.warning("Sube un CSV o cambia a DEMO.")
        st.stop()
    try:
        df = pd.read_csv(file, sep=sep)
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        st.stop()

st.subheader("Vista previa")
st.dataframe(df.head(10), use_container_width=True)

# --------- SELECCIÓN DE COLUMNAS
if df.shape[1] < 2:
    st.error("Necesito al menos dos columnas.")
    st.stop()

cols = list(df.columns)
col_x = st.selectbox("Columna X (independiente)", cols, index=0)
col_y = st.selectbox("Columna Y (dependiente)", cols, index=1 if len(cols) > 1 else 0)

work = df[[col_x, col_y]].copy()

# Convertir a numérico lo que se pueda
for c in [col_x, col_y]:
    work[c] = pd.to_numeric(work[c], errors="coerce")

antes = len(work)
work = work.dropna()
descartadas = antes - len(work)
if descartadas > 0:
    st.info(f"Se descartaron {descartadas} filas no numéricas o vacías.")

if len(work) < 10:
    st.error("Muy pocos datos válidos tras limpiar. Prueba otras columnas o usa DEMO.")
    st.stop()

X = work[[col_x]].to_numpy(dtype=float)
Y = work[col_y].to_numpy(dtype=float)

# --------- ENTRENAMIENTO
test_size = st.slider("Proporción de test", 0.1, 0.5, 0.25, 0.05)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
model = LinearRegression().fit(X_train, Y_train)

# --------- MÉTRICAS
y_pred = model.predict(X_test)
# Evitar parámetro 'squared' por compatibilidad de versiones en la nube
mse = mean_squared_error(Y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = float(r2_score(Y_test, y_pred))
b1 = float(model.coef_[0]); b0 = float(model.intercept_)

st.subheader("Ecuación del modelo")
st.latex(r"y = \beta_0 + \beta_1 x")
st.write(f"**β₀ (intersección):** {b0:,.4f}    |    **β₁ (pendiente):** {b1:,.4f}")

c1, c2 = st.columns(2)
c1.metric("R² (test)", f"{r2:.4f}")
c2.metric("RMSE (test)", f"{rmse:.4f}")

# --------- GRÁFICA
grid = pd.DataFrame({col_x: np.linspace(X.min(), X.max(), 100)})
grid[col_y] = model.predict(grid[[col_x]])

scatter = alt.Chart(work).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X(col_x, title=col_x),
    y=alt.Y(col_y, title=col_y),
    tooltip=[col_x, col_y]
)
line = alt.Chart(grid).mark_line().encode(x=col_x, y=col_y)
st.subheader("Ajuste del modelo")
st.altair_chart(scatter + line, use_container_width=True)

# --------- PREDICCIÓN MANUAL
st.subheader("Predicción rápida")
x_new = st.number_input(f"Nuevo valor para {col_x}", value=float(np.median(X)))
y_new = model.predict(np.array([[x_new]])).item()
st.success(f"Predicción de **{col_y}** para {col_x}={x_new:,.2f}: **{y_new:,.4f}**")