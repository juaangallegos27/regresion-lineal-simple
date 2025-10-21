import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import altair as alt

st.set_page_config(page_title="Regresión lineal simple", page_icon="📈")

st.title("📈 Regresión lineal simple")
st.markdown(
    "Sube un CSV o usa datos de ejemplo. Selecciona X (independiente) e Y (dependiente) "
    "y entreno un modelo lineal en segundos."
)

# Sidebar: carga de datos
st.sidebar.header("Datos")
file = st.sidebar.file_uploader("CSV con encabezados", type=["csv"])
sep = st.sidebar.selectbox("Separador", [",", ";", "\t"], index=0)

if file:
    df = pd.read_csv(file, sep=sep)
else:
    # Datos sintéticos si no suben nada
    rng = np.random.default_rng(42)
    x = np.linspace(0, 100, 120)
    y = 2.5 + 1.7*x + rng.normal(0, 15, size=len(x))
    df = pd.DataFrame({"x": x, "y": y})

st.subheader("Vista previa")
st.dataframe(df.head(10), use_container_width=True)

# Selección de columnas
cols = list(df.columns)
if len(cols) < 2:
    st.stop()
col_x = st.selectbox("Columna X (independiente)", cols, index=0)
col_y = st.selectbox("Columna Y (dependiente)", cols, index=1 if len(cols) > 1 else 0)

data = df[[col_x, col_y]].dropna()
X = data[[col_x]].values.astype(float)
Y = data[col_y].values.astype(float)

# Split
test_size = st.slider("Proporción de test", 0.1, 0.5, 0.2, 0.05)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

# Modelo
model = LinearRegression().fit(X_train, Y_train)

# Métricas
y_pred = model.predict(X_test)
r2 = r2_score(Y_test, y_pred)
rmse = mean_squared_error(Y_test, y_pred, squared=False)

b1 = float(model.coef_[0])
b0 = float(model.intercept_)

st.subheader("Ecuación")
st.latex(r"y = \beta_0 + \beta_1 x")
st.write(f"**β₀ (intersección):** {b0:,.4f}    |    **β₁ (pendiente):** {b1:,.4f}")

c1, c2 = st.columns(2)
c1.metric("R² (test)", f"{r2:.4f}")
c2.metric("RMSE (test)", f"{rmse:.4f}")

# Gráfica
grid = pd.DataFrame({col_x: np.linspace(X.min(), X.max(), 100)})
grid[col_y] = model.predict(grid[[col_x]])

scatter = alt.Chart(data).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X(col_x, title=col_x),
    y=alt.Y(col_y, title=col_y),
    tooltip=[col_x, col_y]
)
line = alt.Chart(grid).mark_line().encode(x=col_x, y=col_y)
st.subheader("Ajuste")
st.altair_chart(scatter + line, use_container_width=True)

# Predicción rápida
st.subheader("Predicción")
x_new = st.number_input(f"Nuevo valor para {col_x}", value=float(np.median(X)))
y_new = model.predict(np.array([[x_new]])).item()
st.write(f"**Predicción de {col_y}:** {y_new:,.4f}")