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

# Nuevo texto visible para confirmar la fuente actual
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

# -------------------- PREVISIÓN Y MÉTRICAS --------------------
st.subheader("Vista previa de los datos actuales")
st.dataframe(df.head(20), use_container_width=True)

test_size = st.slider("Proporción de test", 0.1, 0.5, 0.25, 0.05)
X = df[["hp"]].to_numpy(dtype=float)
Y = df["mpg"].to_numpy(dtype=float)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
model = LinearRegression().fit(X_train, Y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = float(r2_score(Y_test, y_pred))
b1 = float(model.coef_[0])
b0 = float(model.intercept_)

st.subheader("Ecuación del modelo")
st.latex(r"\text{mpg} = \beta_0 + \beta_1 \cdot \text{hp}")
st.write(f"**β₀:** {b0:,.4f} | **β₁:** {b1:,.4f}")

m1, m2 = st.columns(2)
m1.metric("R² (test)", f"{r2:.4f}")
m2.metric("RMSE (test)", f"{rmse:.4f}")

# -------------------- GRÁFICA --------------------
grid = pd.DataFrame({"hp": np.linspace(X.min(), X.max(), 100)})
grid["mpg"] = model.predict(grid[["hp"]])

scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X("hp", title="hp"),
    y=alt.Y("mpg", title="mpg"),
    tooltip=["hp", "mpg"]
)
line = alt.Chart(grid).mark_line().encode(x="hp", y="mpg")

st.subheader("Ajuste del modelo con los datos actuales")
st.altair_chart(scatter + line, use_container_width=True)

# -------------------- PREDICCIÓN --------------------
st.header("Predicción con datos nuevos")
x_new = st.number_input("hp (valor único)", value=float(np.median(X)))
y_new = model.predict(np.array([[x_new]])).item()
st.success(f"Predicción de mpg para hp={x_new:,.2f}: **{y_new:,.4f}**")

st.markdown("Carga un CSV de **predicción** con columna `hp` para obtener `mpg`:")
pred_file = st.file_uploader("CSV para predecir (columna: hp)", type=["csv"], key="pred")
if pred_file is not None:
    try:
        pdf = pd.read_csv(pred_file)
        if "hp" not in pdf.columns:
            st.error("El CSV de predicción debe tener columna 'hp'.")
        else:
            tmp = pdf[["hp"]].copy()
            tmp["hp"] = pd.to_numeric(tmp["hp"], errors="coerce")
            tmp = tmp.dropna()
            if len(tmp) == 0:
                st.error("No hay valores numéricos válidos en 'hp'.")
            else:
                out = tmp.copy()
                out["mpg_pred"] = model.predict(out[["hp"]])
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "Descargar predicciones (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_mpg.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"Error en predicción: {e}")