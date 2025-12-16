import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

# ============================================================
# KONFIGURASI APLIKASI
# ============================================================

st.set_page_config(
    page_title="EEG Emotion Prediction",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# KONSTANTA DATASET
# ============================================================

N_CHANNELS = 14
N_FEATURES = 33
N_CLASSES = 27

# Jumlah fitur SESUAI TRAINING tiap model
MODEL_FEATURES = {
    "CNN": 1,
    "RNN": 28,
    "Hybrid_CNN_LSTM": 28
}

# ============================================================
# DICTIONARY 27 EMOSI (COWEN MODEL)
# ============================================================

EMOTIONS = {
    1: "Admiration", 2: "Adoration", 3: "Aesthetic Appreciation",
    4: "Amusement", 5: "Anger", 6: "Anxiety", 7: "Awe",
    8: "Awkwardness", 9: "Boredom", 10: "Calmness",
    11: "Confusion", 12: "Craving", 13: "Disgust",
    14: "Empathic Pain", 15: "Entrancement", 16: "Excitement",
    17: "Fear", 18: "Horror", 19: "Interest", 20: "Joy",
    21: "Nostalgia", 22: "Relief", 23: "Romance",
    24: "Sadness", 25: "Satisfaction", 26: "Sexual Desire",
    27: "Surprise"
}

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_eeg_model(model_path):
    return load_model(model_path, compile=False)

# ============================================================
# FEATURE COLUMNS
# ============================================================

def get_feature_columns():
    stat_features = [
        'min', 'max', 'ar1', 'ar2', 'ar3', 'ar4', 'md', 'var', 'sd',
        'am', 're', 'le', 'sh', 'te', 'lrssv', 'mte', 'me', 'mcl',
        'n2d', '2d', 'n1d', '1d', 'kurt', 'skew', 'hc', 'hm', 'ha',
        'bpd', 'bpt', 'bpa', 'bpb', 'bpg', 'rba'
    ]

    return [
        f"{stat}_{ch}"
        for ch in range(1, N_CHANNELS + 1)
        for stat in stat_features
    ]

# ============================================================
# PREPROCESSING (MODEL-SPECIFIC)
# ============================================================

def preprocess_data(df, model_type):
    feature_cols = get_feature_columns()

    # Validasi kolom CSV
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Kolom CSV tidak lengkap. Contoh kolom hilang: {list(missing_cols)[:5]}"
        )

    # Ambil fitur → reshape
    X = df[feature_cols].values
    X = X.reshape(-1, N_CHANNELS, N_FEATURES)  # (1, 14, 33)

    # =====================================================
    # CNN
    # =====================================================
    if model_type == "CNN":
        # Global Average Pooling
        X = X.mean(axis=2, keepdims=True)  # (1, 14, 1)

    # =====================================================
    # RNN
    # =====================================================
    elif model_type == "RNN":
        X = X[:, :, :MODEL_FEATURES["RNN"]]  # (1, 14, 28)

    # =====================================================
    # HYBRID CNN + LSTM
    # =====================================================
    elif model_type == "Hybrid_CNN_LSTM":
        X = X[:, :, :MODEL_FEATURES["Hybrid_CNN_LSTM"]]  # (1, 14, 28)

    else:
        raise ValueError("Model type tidak dikenali")

    return X.astype(np.float32)

# ============================================================
# UI STREAMLIT
# ============================================================

st.title("🧠 EEG Emotion Prediction")
st.markdown("Prediksi emosi dari data EEG menggunakan Deep Learning")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Pilih Model:",
    ["CNN", "RNN", "Hybrid_CNN_LSTM"]
)

model_descriptions = {
    "CNN": "Convolutional Neural Network (Conv1D)",
    "RNN": "Recurrent Neural Network (LSTM)",
    "Hybrid_CNN_LSTM": "CNN (Conv1D) + LSTM"
}

st.sidebar.info(f"**{model_descriptions[model_choice]}**")

# ============================================================
# LOAD MODEL
# ============================================================

model_path = f"models/model_{model_choice.lower()}.h5"

if not os.path.exists(model_path):
    st.error(f"❌ Model tidak ditemukan: `{model_path}`")
    st.stop()

with st.spinner(f"Loading {model_choice} model..."):
    model = load_eeg_model(model_path)

st.success(f"✅ Model {model_choice} berhasil dimuat!")

# ============================================================
# UPLOAD & PREDIKSI
# ============================================================

st.markdown("---")
st.header("📁 Upload Data EEG")

st.info(f"""
**Format CSV:**
- 1 baris data
- {N_CHANNELS * N_FEATURES} kolom ({N_CHANNELS} channel × {N_FEATURES} fitur)
- Contoh kolom: `min_1, max_1, ar1_1, ..., rba_14`
""")

uploaded_file = st.file_uploader("Upload file CSV (1 baris)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("❌ File CSV kosong.")
            st.stop()

        if len(df) > 1:
            st.warning("⚠️ Lebih dari 1 baris. Hanya baris pertama yang digunakan.")
            df = df.iloc[[0]]

        st.success(f"✅ File berhasil diupload. Shape: {df.shape}")

        with st.expander("🔍 Preview Data (20 kolom pertama)"):
            st.dataframe(df.iloc[:, :20])

        if st.button("🚀 PREDICT", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                X = preprocess_data(df, model_choice)

                # Validasi dimensi input
                if len(model.input_shape) != len(X.shape):
                    raise ValueError(
                        f"Input shape tidak sesuai. Model expects {model.input_shape}, "
                        f"tapi input {X.shape}"
                    )

                preds = model.predict(X, verbose=0)[0]

                st.markdown("---")
                st.header("🎯 Hasil Prediksi")

                top_idx = np.argmax(preds)
                top_prob = preds[top_idx] * 100

                col1, col2, col3 = st.columns(3)

                col1.metric("Predicted Emotion", EMOTIONS[top_idx + 1], f"{top_prob:.2f}%")

                idx2 = np.argsort(preds)[-2]
                col2.metric("2nd Best", EMOTIONS[idx2 + 1], f"{preds[idx2]*100:.2f}%")

                idx3 = np.argsort(preds)[-3]
                col3.metric("3rd Best", EMOTIONS[idx3 + 1], f"{preds[idx3]*100:.2f}%")

                if top_prob < 30:
                    st.warning("⚠️ Confidence rendah, hasil prediksi kurang meyakinkan.")

                st.markdown("### 📈 Confidence Visualization")
                top5 = np.argsort(preds)[-5:][::-1]

                for idx in top5:
                    st.progress(
                        float(preds[idx]),
                        text=f"{EMOTIONS[idx + 1]}: {preds[idx]*100:.2f}%"
                    )

                with st.expander("📋 Semua 27 Probabilitas"):
                    df_pred = pd.DataFrame({
                        "Emotion": [EMOTIONS[i + 1] for i in range(N_CLASSES)],
                        "Probability (%)": [f"{p*100:.2f}" for p in preds]
                    })
                    st.dataframe(df_pred, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align:center'>
<b>27 Emotion Categories (Cowen Model)</b><br/>
<small>
Admiration • Adoration • Aesthetic • Amusement • Anger • Anxiety • Awe •
Awkwardness • Boredom • Calmness • Confusion • Craving • Disgust •
Empathic Pain • Entrancement • Excitement • Fear • Horror • Interest •
Joy • Nostalgia • Relief • Romance • Sadness • Satisfaction •
Sexual Desire • Surprise
</small>
</div>
""", unsafe_allow_html=True)
