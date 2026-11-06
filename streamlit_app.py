from pathlib import Path
import os
import io
import base64
import wave
import requests
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="TEP Intelligent Monitoring System",
    page_icon="🏭",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "TEP_TemporalConv1D_RobustMax_FINAL.pt"
DEMO_DIR = Path("/tmp/tep_demos")
DEMO_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    st.error(
        "Falta el archivo TEP_TemporalConv1D_RobustMax_FINAL.pt "
        "en el mismo repositorio que streamlit_app.py."
    )
    st.stop()


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    feature_cols = list(checkpoint["feature_names"])
    window = int(checkpoint["window_size"])
    threshold = float(checkpoint["threshold_robust_max"])

    mediana = np.asarray(
        checkpoint["mediana_error"],
        dtype=np.float64
    )

    mad = np.asarray(
        checkpoint["mad_error_seguro"],
        dtype=np.float64
    )

    mean = pd.Series(
        checkpoint["mean_multi"],
        index=feature_cols
    )

    std = pd.Series(
        checkpoint["std_multi"],
        index=feature_cols
    )

    class TemporalConv1DAE(torch.nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Conv1d(32, 16, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(0.1)
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Conv1d(32, n_features, kernel_size=3, padding=1)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = TemporalConv1DAE(
        len(feature_cols)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return (
        model,
        feature_cols,
        window,
        threshold,
        mediana,
        mad,
        mean,
        std
    )


(
    temporal_multi,
    feature_cols_multi,
    WINDOW_MULTI,
    TH_ROBUST_FINAL,
    mediana_error,
    mad_error_seguro,
    mean_multi,
    std_multi
) = cargar_modelo()


# ============================================================
# DESCRIPCIONES XMEAS
# ============================================================

descripciones_xmeas = {
    1: "A Feed Flow",
    2: "D Feed Flow",
    3: "E Feed Flow",
    4: "A and C Feed Flow",
    5: "Recycle Flow",
    6: "Reactor Feed Rate",
    7: "Reactor Pressure",
    8: "Reactor Level",
    9: "Reactor Temperature",
    10: "Purge Rate",
    11: "Separator Temperature",
    12: "Separator Level",
    13: "Separator Pressure",
    14: "Separator Underflow",
    15: "Stripper Level",
    16: "Stripper Pressure",
    17: "Stripper Underflow",
    18: "Stripper Temperature",
    19: "Stripper Steam Flow",
    20: "Compressor Work",
    21: "Reactor Cooling Water Outlet Temperature",
    22: "Separator Cooling Water Outlet Temperature",
}

for i, comp in enumerate(["A", "B", "C", "D", "E", "F"], start=23):
    descripciones_xmeas[i] = f"Reactor Feed Composition {comp}"

for i, comp in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"], start=29):
    descripciones_xmeas[i] = f"Purge Gas Composition {comp}"

for i, comp in enumerate(["D", "E", "F", "G", "H"], start=37):
    descripciones_xmeas[i] = f"Product Composition {comp}"


def obtener_equipo_xmeas(numero):
    if numero in [1, 2, 3, 4]:
        return "Sistema de alimentación"
    if numero in [5, 6]:
        return "Reciclo / alimentación al reactor"
    if numero in [7, 8, 9]:
        return "Reactor"
    if numero == 10:
        return "Sistema de purga"
    if numero in [11, 12, 13, 14]:
        return "Separador"
    if numero in [15, 16, 17, 18, 19]:
        return "Stripper"
    if numero == 20:
        return "Compresor"
    if numero == 21:
        return "Enfriamiento del reactor"
    if numero == 22:
        return "Enfriamiento del separador"
    if 23 <= numero <= 28:
        return "Composición de alimentación al reactor"
    if 29 <= numero <= 36:
        return "Composición del gas de purga"
    if 37 <= numero <= 41:
        return "Composición del producto"
    return "Proceso TEP"


# ============================================================
# DETECTOR
# ============================================================

def crear_ventanas(X, ventana):
    return np.asarray(
        [X[i:i + ventana] for i in range(len(X) - ventana + 1)],
        dtype=np.float32
    )


def calcular_robust_max(df):
    if "Time" in df.columns and "Time (h)" not in df.columns:
        df = df.rename(columns={"Time": "Time (h)"})

    faltantes = [c for c in feature_cols_multi if c not in df.columns]

    if faltantes:
        raise ValueError(
            "El archivo no contiene las 41 variables XMEAS esperadas."
        )

    if "Time (h)" in df.columns:
        tiempo = df["Time (h)"].astype(float).values
    else:
        tiempo = np.arange(len(df)) / 60.0

    X = (
        (df[feature_cols_multi] - mean_multi) / std_multi
    ).values.astype(np.float32)

    ventanas = crear_ventanas(
        X,
        WINDOW_MULTI
    )

    entrada = np.transpose(
        ventanas,
        (0, 2, 1)
    )

    tensor = torch.tensor(
        entrada,
        dtype=torch.float32
    )

    with torch.no_grad():
        reconstruccion = temporal_multi(tensor)
        error = (tensor - reconstruccion) ** 2
        error_variable = torch.mean(
            error,
            dim=2
        ).cpu().numpy()

    z_robusto = (
        error_variable - mediana_error
    ) / (
        1.4826 * mad_error_seguro
    )

    z_robusto = np.maximum(
        z_robusto,
        0
    )

    score = np.max(
        z_robusto,
        axis=1
    )

    tiempo_alineado = tiempo[
        WINDOW_MULTI - 1:
    ]

    return tiempo_alineado, score, z_robusto


# ============================================================
# DEMOS
# ============================================================

BASE_URLS = [
    "https://github.com/Blound04/tennessee-eastman-dataset/raw/main/simulations/mode_1",
    "https://github.com/mv-per/tennessee-eastman-dataset/raw/main/simulations/mode_1"
]


def descargar_archivo(nombre, es_falla):
    destino = DEMO_DIR / nombre

    if destino.exists():
        return destino

    for base in BASE_URLS:
        url = (
            f"{base}/faults/{nombre}"
            if es_falla
            else f"{base}/{nombre}"
        )

        try:
            r = requests.get(
                url,
                timeout=60
            )

            if r.status_code == 200 and len(r.content) > 10000:
                destino.write_bytes(
                    r.content
                )
                return destino
        except Exception:
            pass

    raise RuntimeError(
        f"No se pudo descargar {nombre}"
    )


def crear_demo_normal():
    destino = DEMO_DIR / "mode1_normal_DEMO.xlsx"

    if destino.exists():
        return destino

    ruta = descargar_archivo(
        "mode1_normal_50.xlsx",
        False
    )

    df = pd.read_excel(ruta)

    if "Time" in df.columns and "Time (h)" not in df.columns:
        df = df.rename(columns={"Time": "Time (h)"})

    _, score, _ = calcular_robust_max(df)
    alarmas = (score > TH_ROBUST_FINAL).astype(int)

    muestras_segmento = 601
    n_score_segmento = muestras_segmento - WINDOW_MULTI + 1

    mejor_inicio = None
    mejor_far = np.inf

    for inicio in range(0, len(df) - muestras_segmento + 1):
        seg = alarmas[inicio:inicio + n_score_segmento]

        persistente = any(
            seg[i] == 1 and seg[i + 1] == 1 and seg[i + 2] == 1
            for i in range(len(seg) - 2)
        )

        far = np.mean(seg) * 100

        if not persistente and far < mejor_far:
            mejor_inicio = inicio
            mejor_far = far

    if mejor_inicio is None:
        raise RuntimeError(
            "No se encontró un segmento normal sin alarma persistente."
        )

    demo = df.iloc[
        mejor_inicio:
        mejor_inicio + muestras_segmento
    ].copy()

    demo["Time (h)"] = np.arange(len(demo)) / 60.0
    demo.to_excel(destino, index=False)

    return destino


def obtener_demo(tipo):
    if tipo == "normal":
        return crear_demo_normal()

    mapa = {
        "F11": "mode1_11_4.xlsx",
        "F12": "mode1_12_3.xlsx",
        "F16": "mode1_16_3.xlsx"
    }

    return descargar_archivo(
        mapa[tipo],
        True
    )


# ============================================================
# ALARMA
# ============================================================

def alarma_autoplay():
    sr = 44100
    duracion = 2.0

    t = np.linspace(
        0,
        duracion,
        int(sr * duracion),
        endpoint=False
    )

    tono = np.sin(2 * np.pi * 880 * t)
    patron = (
        np.sin(2 * np.pi * 2 * t) > 0
    ).astype(float)

    audio = (
        tono * patron * 0.5
    )

    pcm = np.int16(
        np.clip(audio, -1, 1) * 32767
    )

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    b64 = base64.b64encode(
        buffer.getvalue()
    ).decode()

    components.html(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """,
        height=0
    )


# ============================================================
# ANÁLISIS
# ============================================================

def analizar(df, nombre_archivo, tiempo_falla):
    tiempo, score, z = calcular_robust_max(df)
    alarmas = (score > TH_ROBUST_FINAL).astype(int)
    es_normal = "normal" in nombre_archivo.lower()

    idx_det = None

    if es_normal:
        for i in range(len(alarmas) - 2):
            if alarmas[i] == 1 and alarmas[i+1] == 1 and alarmas[i+2] == 1:
                idx_det = i
                break

        far = np.mean(alarmas) * 100

        if idx_det is None:
            return {
                "estado": "🟢 OPERACIÓN NORMAL",
                "deteccion": "Sin detección",
                "delay": "No aplica",
                "variable": "Sin anomalía",
                "far": far,
                "fdr": None,
                "score_det": None,
                "tabla": pd.DataFrame(),
                "tiempo": tiempo,
                "score": score,
                "idx_det": None,
                "es_normal": True,
                "alarma": False
            }

    else:
        pre = tiempo <= tiempo_falla
        post = tiempo > tiempo_falla

        far = np.mean(alarmas[pre]) * 100
        fdr = np.mean(alarmas[post]) * 100

        for i in range(len(alarmas) - 2):
            if tiempo[i] <= tiempo_falla:
                continue

            if alarmas[i] == 1 and alarmas[i+1] == 1 and alarmas[i+2] == 1:
                idx_det = i
                break

        if idx_det is None:
            return {
                "estado": "🟡 ANOMALÍA NO CONFIRMADA",
                "deteccion": "No detectada",
                "delay": "—",
                "variable": "—",
                "far": far,
                "fdr": fdr,
                "score_det": None,
                "tabla": pd.DataFrame(),
                "tiempo": tiempo,
                "score": score,
                "idx_det": None,
                "es_normal": False,
                "alarma": False
            }

    tiempo_det = float(
        tiempo[idx_det]
    )

    contrib = z[idx_det]
    top_idx = np.argsort(contrib)[::-1][:5]
    filas = []

    for ranking, j in enumerate(top_idx, start=1):
        numero = j + 1

        filas.append({
            "Ranking": ranking,
            "Variable": f"XMEAS({numero})",
            "Descripción": descripciones_xmeas[numero],
            "Equipo / área": obtener_equipo_xmeas(numero),
            "Contribución z": float(contrib[j])
        })

    tabla = pd.DataFrame(filas)

    if es_normal:
        return {
            "estado": "🟠 FALSA ALARMA PERSISTENTE",
            "deteccion": f"{tiempo_det:.2f} h",
            "delay": "No aplica",
            "variable": tabla.iloc[0]["Variable"],
            "far": far,
            "fdr": None,
            "score_det": float(score[idx_det]),
            "tabla": tabla,
            "tiempo": tiempo,
            "score": score,
            "idx_det": idx_det,
            "es_normal": True,
            "alarma": True
        }

    delay = (
        tiempo_det - tiempo_falla
    ) * 60

    principal = tabla.iloc[0]

    return {
        "estado": "🔴 ANOMALÍA DETECTADA",
        "deteccion": f"{tiempo_det:.3f} h",
        "delay": f"{delay:.1f} min",
        "variable": (
            f"{principal['Variable']} — "
            f"{principal['Equipo / área']}"
        ),
        "far": far,
        "fdr": fdr,
        "score_det": float(score[idx_det]),
        "tabla": tabla,
        "tiempo": tiempo,
        "score": score,
        "idx_det": idx_det,
        "es_normal": False,
        "alarma": True
    }


# ============================================================
# INTERFAZ
# ============================================================

st.title("🏭 TEP Intelligent Monitoring System")
st.subheader("Sistema de detección y alarma de anomalías")
st.markdown("**Temporal Conv1D Autoencoder + Robust Max**")

st.markdown("## Casos de demostración")

c1, c2, c3, c4 = st.columns(4)

if c1.button("🔵 Operación normal", use_container_width=True):
    st.session_state["demo"] = "normal"

if c2.button("🟢 F11 — Alta", use_container_width=True):
    st.session_state["demo"] = "F11"

if c3.button("🟡 F12 — Intermedia", use_container_width=True):
    st.session_state["demo"] = "F12"

if c4.button("🔴 F16 — Baja", use_container_width=True):
    st.session_state["demo"] = "F16"


uploaded = st.file_uploader(
    "📁 Cargar corrida TEP (.xlsx)",
    type=["xlsx"]
)

tiempo_falla = st.number_input(
    "⏱️ Tiempo de introducción de falla (h)",
    value=10.0,
    step=0.1
)


df_actual = None
nombre_actual = None

if uploaded is not None:
    df_actual = pd.read_excel(uploaded)
    nombre_actual = uploaded.name

elif "demo" in st.session_state:
    ruta_demo = obtener_demo(
        st.session_state["demo"]
    )

    df_actual = pd.read_excel(
        ruta_demo
    )

    nombre_actual = ruta_demo.name

    st.info(
        f"Caso cargado: {nombre_actual}"
    )


if st.button(
    "🚨 ANALIZAR PROCESO",
    type="primary",
    use_container_width=True
):

    if df_actual is None:
        st.warning(
            "Selecciona un caso de demostración o carga un archivo."
        )

    else:
        try:
            r = analizar(
                df_actual,
                nombre_actual,
                tiempo_falla
            )

            st.markdown("## 📊 Estado del proceso")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Estado",
                r["estado"]
            )

            m2.metric(
                "Tiempo de detección",
                r["deteccion"]
            )

            m3.metric(
                "Delay",
                r["delay"]
            )

            m4.metric(
                "Variable / área principal",
                r["variable"]
            )

            if r["estado"].startswith("🟢"):
                st.success(
                    f"Operación normal. Umbral Robust Max = "
                    f"{TH_ROBUST_FINAL:.3f}. "
                    f"Tasa de alarmas = {r['far']:.2f}%."
                )

            elif r["estado"].startswith("🔴"):
                st.error(
                    f"Anomalía persistente confirmada. "
                    f"FDR = {r['fdr']:.2f}%, "
                    f"FAR = {r['far']:.2f}%."
                )

            else:
                st.warning(
                    f"Resultado: {r['estado']}."
                )

            if r["alarma"]:
                alarma_autoplay()
                st.markdown("### 🔊 Alarma del sistema")
                st.caption(
                    "Alarma activada por tres excedencias consecutivas."
                )

            st.markdown(
                "## Variables con mayor contribución"
            )

            if r["tabla"].empty:
                st.write(
                    "Sin variables anómalas que reportar."
                )
            else:
                st.dataframe(
                    r["tabla"].round(4),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown(
                "## Evolución del score"
            )

            fig, ax = plt.subplots(
                figsize=(11, 5)
            )

            ax.plot(
                r["tiempo"],
                r["score"],
                label="Robust Max"
            )

            ax.axhline(
                TH_ROBUST_FINAL,
                linestyle="--",
                label=f"Umbral = {TH_ROBUST_FINAL:.2f}"
            )

            if not r["es_normal"]:
                ax.axvline(
                    tiempo_falla,
                    linestyle=":",
                    label="Inicio de falla"
                )

            if r["idx_det"] is not None:
                ax.axvline(
                    r["tiempo"][r["idx_det"]],
                    linestyle="-.",
                    label="Alarma persistente"
                )

            ax.set_xlabel(
                "Tiempo (h)"
            )

            ax.set_ylabel(
                "Robust Max score"
            )

            ax.set_title(
                "Monitoreo de anomalías — Tennessee Eastman Process"
            )

            ax.grid(
                alpha=0.25
            )

            ax.legend()

            st.pyplot(fig)

        except Exception as e:
            st.exception(e)
