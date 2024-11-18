import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# Load the model
model = joblib.load('RFC4.pkl')
scaler = joblib.load('scaler4.pkl')

# Define feature options
Level_of_Education_options = {
    0: 'Primary (0)',
    1: 'Secondary (1)',
    2: 'Certificate (2)',
    3: 'Diploma (3)',
    4: 'Degree (4)'
}

Tumor_Grade_options = {
    1: 'Grade1 (1)',
    2: 'Grade2 (2)',
    3: 'Grade3 (3)',
    4: 'Grade4 (4)'
}

# Define feature names
feature_names = [
    "Level_of_Education", "Tumor_Size_2_Years_after_Surgery", "Tumor_Grade",
    "Lymph_Node_Metastasis", "Numbe_of_Lymph_Nodes", "Marital_Status_Unmarried",
    "Marital_Status_Married", "Marital_Status_Divorced"
]

# Streamlit user interface
st.title("Breast Cancer Recurrence Predictor")

# Input features
Level_of_Education = st.selectbox("Level of Education:", options=list(Level_of_Education_options.keys()), format_func=lambda x: Level_of_Education_options[x])
Tumor_Size_2_Years_after_Surgery = st.number_input("Tumor Size 2 Years after Surgery (mm):", min_value=0, max_value=100, value=50)
Tumor_Grade = st.selectbox("Tumor Grade:", options=list(Tumor_Grade_options.keys()), format_func=lambda x: Tumor_Grade_options[x])
Lymph_Node_Metastasis = st.selectbox("Lymph Node Metastasis:", options=['No (0)', 'Yes (1)'], index=0)
Numbe_of_Lymph_Nodes = st.number_input("Number of Lymph Nodes:", min_value=0, max_value=50, value=25)
Marital_Status_Unmarried = st.radio("Marital Status Unmarried:", options=['No (0)', 'Yes (1)'], index=0)
Marital_Status_Married = st.radio("Marital Status Married:", options=['No (0)', 'Yes (1)'], index=0)
Marital_Status_Divorced = st.radio("Marital Status Divorced:", options=['No (0)', 'Yes (1)'], index=0)

# Convert categorical responses to integers
Lymph_Node_Metastasis = 1 if Lymph_Node_Metastasis == 'Yes (1)' else 0
Marital_Status_Unmarried = 1 if Marital_Status_Unmarried == 'Yes (1)' else 0
Marital_Status_Married = 1 if Marital_Status_Married == 'Yes (1)' else 0
Marital_Status_Divorced = 1 if Marital_Status_Divorced == 'Yes (1)' else 0

# Prepare data for prediction
continuous_features = [Tumor_Size_2_Years_after_Surgery, Numbe_of_Lymph_Nodes]
categorical_features = [Level_of_Education, Tumor_Grade, Lymph_Node_Metastasis, Marital_Status_Unmarried, Marital_Status_Married, Marital_Status_Divorced]

# Scale continuous features
continuous_features_scaled = scaler.transform(np.array(continuous_features).reshape(1, -1))

# Combine for prediction
features_for_prediction = np.concatenate([continuous_features_scaled, np.array(categorical_features).reshape(1, -1)], axis=1)

if st.button("Predict"):
    # Predict
    predicted_class = model.predict(features_for_prediction)[0]
    predicted_proba = model.predict_proba(features_for_prediction)[0]

    # Display results
    st.write(f"**Predicted Class**: {predicted_class}(1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities**: {predicted_proba}")

    # Advice
    probability = predicted_proba[predicted_class] * 100
    advice = (
        f"According to our model, you have a {'high' if predicted_class == 1 else 'low'} risk of breast cancer recurrence. "
        f"The model predicts that your probability of {'having' if predicted_class == 1 else 'not having'} breast cancer recurrence is {probability:.1f}%. "
        f"{'It is advised to consult with your healthcare provider for further evaluation and possible intervention.' if predicted_class == 1 else 'Maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider.'}"
    )
    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame(features_for_prediction, columns=feature_names))

    # Display the SHAP force plot for the predicted class
    fig, ax = plt.subplots()
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:, :, 1], pd.DataFrame(features_for_prediction, columns=feature_names), ax=ax)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:, :, 0], pd.DataFrame(features_for_prediction, columns=feature_names), ax=ax)

    # Save plot to BytesIO and display in Streamlit
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=1200)
    buf.seek(0)
    st.image(buf, caption='SHAP Force Plot Explanation', use_column_width=True)
    plt.close(fig)  # Close plot to prevent memory leak