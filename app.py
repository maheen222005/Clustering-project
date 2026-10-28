import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------
# 1. Locate the project files
# ----------------------------------
HERE = Path(__file__).resolve().parent

# ----------------------------------
# 2. Load the trained model and preprocessing files
# ----------------------------------
try:
    imputer = joblib.load(HERE / "imputer.pkl")
    scaler = joblib.load(HERE / "scaler.pkl")
    pca = joblib.load(HERE / "pca.pkl")
    model = joblib.load(HERE / "kmeans_model.pkl")
    feature_names = joblib.load(HERE / "feature_names.pkl")

except FileNotFoundError as e:
    st.error(f"Required project file is missing: {e}")
    st.stop()

# ----------------------------------
# 3. Configure the Streamlit page
# ----------------------------------
st.set_page_config(
    page_title="World Development Clustering",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 World Development Clustering")

st.write(
    "Enter the country development indicators below "
    "to predict the country's cluster."
)

st.info(
    "Enter values using the same units and realistic ranges "
    "as the training dataset."
)

# ----------------------------------
# 4. Collect user input
# ----------------------------------
values = {}

with st.form("cluster_form"):

    st.subheader("Enter Development Indicators")

    for feature in feature_names:
        values[feature] = st.number_input(
            label=str(feature),
            value=0.0,
            format="%.4f",
            help=f"Enter a value for {feature} in the training dataset's units."
        )

    submitted = st.form_submit_button("Predict Cluster")

# ----------------------------------
# 5. Predict the cluster
# ----------------------------------
if submitted:
    try:
        # Convert input into a DataFrame
        input_df = pd.DataFrame(
            [values],
            columns=feature_names
        )

        # Apply preprocessing
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        input_pca = pca.transform(input_scaled)

        # Predict cluster
        cluster = int(model.predict(input_pca)[0])

        # Display prediction
        st.success(f"Predicted Cluster: {cluster}")

        st.write(
            "The country belongs to the cluster identified by "
            f"label **{cluster}**."
        )

        # Debugging information
        st.subheader("Debugging Information")

        st.write("Input values:")
        st.dataframe(input_df)

        st.write("PCA values:")
        st.write(input_pca)

        st.write("Cluster centers:")
        st.write(model.cluster_centers_)

        st.write("Distances from each cluster:")

        distances = model.transform(input_pca)

        st.dataframe(
            pd.DataFrame(
                distances,
                columns=[
                    f"Cluster {i}"
                    for i in range(model.n_clusters)
                ]
            )
        )

        st.warning(
            "Cluster numbers are labels, not rankings. "
            "Use the cluster profiles from your notebook "
            "to explain each group's characteristics."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

