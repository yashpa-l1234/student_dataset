
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="KNN Prediction App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 50%, #e0f2fe 100%);
    color: #172554;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #312e81, #2563eb);
    padding: 35px 30px;
    border-radius: 22px;
    color: white;
    text-align: center;
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.25);
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 10px;
}

.hero p {
    color: #dbeafe;
    font-size: 16px;
}

.card {
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #dbeafe;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.09);
    margin-bottom: 20px;
}

.result-card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.16);
    text-align: center;
    margin-top: 22px;
}

.result-card h2 {
    color: #1d4ed8;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3.2em;
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border: none;
    font-weight: 700;
    font-size: 17px;
    box-shadow: 0 5px 15px rgba(37, 99, 235, 0.25);
    transition: 0.2s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #3730a3);
    color: white;
    transform: translateY(-2px);
}

[data-testid="stSidebar"] {
    background: #eef2ff;
    border-right: 1px solid #dbeafe;
}

label {
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("kn.pkl", "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load kn.pkl: {e}")
    st.stop()


# ---------------- MODEL FEATURES ----------------
feature_names = getattr(model, "feature_names_in_", None)
n_features = getattr(model, "n_features_in_", None)

# Support a fitted preprocessing pipeline where possible
if n_features is None and hasattr(model, "steps"):
    final_estimator = model.steps[-1][1]
    n_features = getattr(final_estimator, "n_features_in_", None)

if feature_names is None and n_features is None:
    st.error(
        "Could not detect the number of input features. "
        "Please check how kn.pkl was trained."
    )
    st.stop()

if feature_names is not None:
    feature_names = list(feature_names)
    n_features = len(feature_names)
else:
    feature_names = [f"Feature {i + 1}" for i in range(n_features)]


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🤖 KNN Prediction Studio</h1>
    <p>Machine Learning • Smart Predictions • Interactive Dashboard</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎯 Model Information")
    st.write("**Algorithm:** K-Nearest Neighbors")
    st.write(f"**Input Features:** {n_features}")
    st.divider()
    st.info(
        "Enter the values required by your trained model "
        "and click Predict."
    )
    st.caption("Built with Python and Streamlit")


# ---------------- INPUT SECTION ----------------
st.markdown("## 📊 Enter Input Details")
st.write("Fill in all feature values used during model training.")

with st.form("prediction_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)

    columns = st.columns(2)
    input_values = []

    for i, feature in enumerate(feature_names):
        with columns[i % 2]:
            value = st.number_input(
                str(feature),
                value=0.0,
                step=1.0,
                format="%.4f",
                key=f"feature_{i}"
            )
            input_values.append(value)

    st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🚀 Predict Now")


# ---------------- PREDICTION ----------------
if submitted:
    try:
        input_df = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        # Preserve feature names when the model was trained with a DataFrame.
        if hasattr(model, "feature_names_in_"):
            prediction_input = input_df
        else:
            prediction_input = np.array([input_values])

        prediction = model.predict(prediction_input)[0]

        st.markdown(f"""
        <div class="result-card">
            <h2>✅ Prediction Result</h2>
            <h1>{prediction}</h1>
            <p>Your model has generated a prediction successfully.</p>
        </div>
        """, unsafe_allow_html=True)

        # Show confidence when supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(prediction_input)[0]
            classes = getattr(model, "classes_", None)

            if classes is not None:
                best_index = int(np.argmax(probabilities))
                confidence = float(probabilities[best_index]) * 100

                st.metric("Prediction Confidence", f"{confidence:.2f}%")

                probability_df = pd.DataFrame({
                    "Class": classes,
                    "Probability (%)": probabilities * 100
                })

                st.markdown("### 📈 Class Probabilities")
                st.bar_chart(
                    probability_df.set_index("Class")
                )

        with st.expander("View submitted input values"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.warning(
            "Check the feature order, data types, and preprocessing "
            "used during model training."
        )


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("KNN Prediction Studio | Powered by Streamlit")
