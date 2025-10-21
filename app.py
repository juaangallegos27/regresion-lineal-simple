import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import altair as alt
from io import StringIO

st.set_page_config(page_title="Autos: mpg vs hp", page_icon="🚗", layout="centered")

st.title("🚗 Regresión lineal simple: consumo (mpg) según potencia (hp)")
st.write(
    "Evaluamos un caso sencillo y realista de autos: **predecir el consumo** en millas por galón "
    "(**mpg**, variable dependiente) a partir de la **potencia del motor** en caballos de fuerza "
    "(**hp**, variable independiente)."
)

# -------------------- FUENTE DE DATOS --------------------
modo = st.radio("Fuente de datos para ENTRENAR el modelo", ["DEMO: autos (recomendado)", "Mi CSV"], horizontal=True)

if modo.startswith("DEMO"):
    st.sidebar.header("Datos DEMO de autos")
    n = st.sidebar.slider("Número de autos", 50, 500, 160, 10)
    beta0_true = st.sidebar.slider("β₀ (mpg base)", 10.0, 60.0, 50.0, 0.5)
    beta1_true = st.sidebar.slider("β₁ (pendiente, mpg por hp)", -0.50, -0.01, -0.05, 0.01)
    ruido = st.sidebar.slider("Ruido (σ mpg)", 0.0, 10.0, 4.0, 0.5)

    rng = np.random.default_rng(42)
    hp = np.linspace(50, 300, n)
    mpg = beta0_true + beta1_true * hp + rng.normal(0, ruido, size=n)
    df = pd.DataFrame({"hp": hp, "mpg": mpg})
    x_col_default, y_col_default = "hp", "mpg"
else:
    st.sidebar.header("Carga tu CSV para ENTRENAR")
    train_file = st.sidebar.file_uploader("CSV con encabezados", type=["csv"])
    sep = st.sidebar.selectbox("Separador", [",", ";", "\t"], index=0)
    if not train_file:
        st.warning("Sube un CSV o usa DEMO.")
        st.stop()
    try:
        df = pd.read_csv(train_file, sep=sep)
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        st.stop()
    x_col_default = df.columns[0]
    y_col_default = df.columns[1] if len(df.columns) > 1 else df.columns[0]

st.subheader("Vista previa de datos para ENTRENAR")
st.dataframe(df.head(10), use_container_width=True)

if df.shape[1] < 2:
    st.error("Necesito al menos dos columnas.")
    st.stop()

cols = list(df.columns)
col_x = st.selectbox("Columna X (independiente, potencia hp)", cols, index=cols.index(x_col_default) if x_col_default in cols else 0)
col_y = st.selectbox("Columna Y (dependiente, consumo mpg)", cols, index=cols.index(y_col_default) if y_col_default in cols else 1)

work = df[[col_x, col_y]].copy()
for c in [col_x, col_y]:
    work[c] = pd.to_numeric(work[c], errors="coerce")
antes = len(work)
work = work.dropna()
descartadas = antes - len(work)
if descartadas > 0:
    st.info(f"Se descartaron {descartadas} filas no numéricas o vacías.")

if len(work) < 20:
    st.error("Muy pocos datos válidos tras limpiar. Revisa las columnas o la fuente de datos.")
    st.stop()

# -------------------- ENTRENAMIENTO --------------------
test_size = st.slider("Proporción de test", 0.1, 0.5, 0.25, 0.05)
X = work[[col_x]].to_numpy(dtype=float)
Y = work[col_y].to_numpy(dtype=float)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
model = LinearRegression().fit(X_train, Y_train)

# Métricas
y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = float(r2_score(Y_test, y_pred))
b1 = float(model.coef_[0])
b0 = float(model.intercept_)

st.subheader("Ecuación del modelo")
st.latex(r"mpg = \beta_0 + \beta_1 \cdot hp")
st.write(f"**β₀:** {b0:,.4f}    |    **β₁:** {b1:,.4f}")
c1, c2 = st.columns(2)
c1.metric("R² (test)", f"{r2:.4f}")
c2.metric("RMSE (test)", f"{rmse:.4f}")

# Gráfica
grid = pd.DataFrame({col_x: np.linspace(X.min(), X.max(), 100)})
grid[col_y] = model.predict(grid[[col_x]])

scatter = alt.Chart(work).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X(col_x, title=f"{col_x} (hp)"),
    y=alt.Y(col_y, title=f"{col_y} (mpg)"),
    tooltip=[col_x, col_y]
)
line = alt.Chart(grid).mark_line().encode(x=col_x, y=col_y)
st.subheader("Ajuste del modelo")
st.altair_chart(scatter + line, use_container_width=True)

# -------------------- PREDICCIONES --------------------
st.header("Predicciones con datos NUEVOS")

st.markdown("### A) Valor único (escribe a mano)")
default_x = float(np.median(X))
x_new = st.number_input(f"Nuevo valor para {col_x} (hp)", value=default_x)
y_new = model.predict(np.array([[x_new]])).item()
st.success(f"Predicción de **{col_y} (mpg)** para **{col_x}={x_new:,.2f}** → **{y_new:,.4f} mpg**")

st.markdown("### B) Lista manual (varios hp)")
raw = st.text_area("Escribe valores de hp separados por comas o saltos de línea", value="100, 150, 200")
if raw.strip():
    try:
        # parsea números
        vals = []
        for token in raw.replace("\n", ",").split(","):
            t = token.strip()
            if t:
                vals.append(float(t))
        batch_df = pd.DataFrame({col_x: vals})
        batch_df[col_y] = model.predict(batch_df[[col_x]])
        st.dataframe(batch_df, use_container_width=True)
        csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar predicciones CSV", data=csv_bytes, file_name="predicciones.csv", mime="text/csv")
    except Exception as e:
        st.error(f"No pude procesar la lista: {e}")

st.markdown("### C) Subir CSV de PREDICCIÓN")
pred_file = st.file_uploader("CSV con una columna de hp (o selecciona cuál es X más abajo)", type=["csv"], key="pred_csv")
pred_sep = st.selectbox("Separador del CSV de predicción", [",", ";", "\t"], index=0, key="pred_sep")

if pred_file is not None:
    try:
        pred_df = pd.read_csv(pred_file, sep=pred_sep)
        st.caption("Vista previa del CSV de predicción")
        st.dataframe(pred_df.head(8), use_container_width=True)

        pred_cols = list(pred_df.columns)
        pred_x_col = st.selectbox("¿Qué columna usar como hp (X) para predecir mpg?", pred_cols, index=0, key="pred_x_col")

        tmp = pred_df[[pred_x_col]].copy()
        tmp[pred_x_col] = pd.to_numeric(tmp[pred_x_col], errors="coerce")
        tmp = tmp.dropna()
        if len(tmp) == 0:
            st.error("No hay valores numéricos válidos en esa columna.")
        else:
            out = tmp.rename(columns={pred_x_col: col_x})
            out[col_y] = model.predict(out[[col_x]])
            st.dataframe(out, use_container_width=True)
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar predicciones CSV", data=csv_bytes, file_name="predicciones.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error al leer o procesar el CSV de predicción: {e}")