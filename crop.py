import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the App
st.title("Crop Prediction and Analysis App 🌾")

# Load internal dataset
@st.cache_data
def load_data():
    return pd.read_csv("crop.csv")

# Save updated dataset
def save_data(updated_data):
    updated_data.to_csv("crop.csv", index=False)

data = load_data()

# Display dataset preview
st.write("### Dataset Preview:")
st.write(data.head())

# Update dataset functionality
st.write("### Do you want to update the dataset?")
update_choice = st.radio("Select an option:", ("No", "Yes"), index=0)

if update_choice == "Yes":
    st.write("### Add New Data")
    # Create placeholders for new data
    col1, col2, col3 = st.columns(3)
    with col1:
        new_crop = st.text_input("Crop Name", "")
    with col2:
        new_nitrogen = st.number_input("Nitrogen (N)", value=0.0)
    with col3:
        new_phosphorus = st.number_input("Phosphorus (P)", value=0.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        new_potassium = st.number_input("Potassium (K)", value=0.0)
    with col5:
        new_ph = st.number_input("pH", value=7.0)
    with col6:
        new_rainfall = st.number_input("Rainfall (mm)", value=0.0)

    temperature = st.number_input("Temperature (°C)", value=25.0)

    # Append new data
    if st.button("Update Dataset"):
        if new_crop.strip() == "":
            st.error("Crop name cannot be empty!")
        else:
            new_row = {
                "Crop": new_crop,
                "N": new_nitrogen,
                "P": new_phosphorus,
                "K": new_potassium,
                "pH": new_ph,
                "rainfall": new_rainfall,
                "temperature": temperature,
            }
            # Convert the new row to a DataFrame and append to the dataset
            new_row_df = pd.DataFrame([new_row])
            updated_data = pd.concat([data, new_row_df], ignore_index=True)
            save_data(updated_data)  # Save the updated dataset to CSV
            st.success("Dataset updated successfully! Reload the app to see changes.")

# Label encode the 'Crop' column
if "Crop" in data.columns:
    le = LabelEncoder()
    data["Crop"] = le.fit_transform(data["Crop"])  # Encode crop names into integers

# Display heatmap of the dataset
st.write("### Dataset Correlation Heatmap:")
numeric_data = data.select_dtypes(include=[np.number])  # Select numeric columns only
fig, ax = plt.subplots()
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Train model section
st.header("Step 2: Train the Model")
if st.button("Train"):
    if "Crop" not in data.columns:
        st.error("The dataset must contain a 'Crop' column as the target variable.")
    else:
        # Separate features and target
        X = data.drop(columns=["Crop"])
        y = data["Crop"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Decision Tree model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Test the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Display test accuracy
        st.success(f"Test Accuracy: {accuracy:.2f}")

        # Save the trained model for later use
        st.session_state["model"] = model
        st.session_state["label_encoder"] = le

# Prediction section
if "model" in st.session_state:
    st.header("Step 3: Predict the Crop")
    st.write("Enter the feature values below:")

    # Dynamically generate input fields based on feature columns
    features = [col for col in data.columns if col != "Crop"]
    user_input = {}

    # Create columns for a better layout
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            user_input[feature] = st.number_input(f"{feature.title()}", value=0.0, help=f"Enter the value for {feature}")

    if st.button("Predict"):
        # Prepare input for prediction
        input_values = np.array([list(user_input.values())]).reshape(1, -1)
        prediction_encoded = st.session_state["model"].predict(input_values)[0]
        prediction = st.session_state["label_encoder"].inverse_transform([prediction_encoded])[0]
        st.success(f"The predicted crop is: **{prediction.title()}** 🌱")
