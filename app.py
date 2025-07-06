import streamlit as st
import joblib
import pandas as pd

st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements below:")

# Input sliders
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Load model
#model = joblib.load("iris_model1.pkl")

# Load model and accuracy
model_data = joblib.load("iris_model.pkl")
model = model_data['model']
accuracy = model_data['accuracy']


# Prepare input as DataFrame to avoid warnings
input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

# Predict
prediction = model.predict(input_df)[0]

# Show result
species = ["Setosa", "Versicolor", "Virginica"]
st.success(f"Predicted species: **{species[prediction]}** \nModel Accuracy: **{accuracy * 100:.2f}%**")

