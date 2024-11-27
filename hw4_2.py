import pandas as pd
from pycaret.classification import *
import streamlit as st

# Feature Engineering and Preprocessing
def preprocess_data(data):
    # Drop irrelevant columns
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    return data

# Streamlit Interface
st.title("Model Optimization with PyCaret")

uploaded_file = st.file_uploader("Upload Titanic Dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)

    # Preprocess data
    data = preprocess_data(data)
    st.write("### Data After Preprocessing")
    st.dataframe(data.head())

    # PyCaret Setup
    st.write("### Setting up PyCaret Environment")
    with st.spinner("Setting up PyCaret..."):
        # Remove the 'silent' parameter
        clf_setup = setup(data=data, target='Survived', session_id=123)
        st.success("PyCaret Environment Setup Complete")

    # Compare Models
    st.write("### Comparing Models")
    with st.spinner("Comparing models..."):
        best_model = compare_models()
    st.write("Best Model Selected:", best_model)

    # Tune Best Model
    st.write("### Tuning Best Model")
    with st.spinner("Optimizing hyperparameters..."):
        tuned_model = tune_model(best_model)
    st.write("Tuned Model:", tuned_model)

    # Finalize the Model
    st.write("### Training Final Model")
    final_model = finalize_model(tuned_model)

    # Predict with Final Model
    st.write("### Predictions on Test Data")
    predictions = predict_model(final_model)
    st.dataframe(predictions)
else:
    st.info("Please upload a dataset to proceed.")
