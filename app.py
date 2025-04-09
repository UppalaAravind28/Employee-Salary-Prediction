import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Custom CSS for professional styling
st.markdown("""
    <style>
  /* Reset & Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background-color: #f4f6f9;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #2c3e50;
    line-height: 1.6;
}

/* Header */
.header {
    background-color: #2c3e50;
    color: #ecf0f1;
    padding: 30px 20px;
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    letter-spacing: 1px;
    border-bottom: 4px solid #2980b9;
}

/* Navbar */
.navbar {
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #34495e;
    padding: 12px 0;
    gap: 20px;
    flex-wrap: wrap;
}

.navbar a {
    color: #ffffff;
    padding: 10px 18px;
    text-decoration: none;
    font-size: 16px;
    font-weight: 500;
    border-radius: 6px;
    transition: all 0.3s ease;
}

.navbar a:hover,
.navbar a.active {
    background-color: #2980b9;
}

/* Sidebar */
.sidebar {
    background-color: #ffffff;
    color: #2c3e50; /* Ensuring dark text on light background */
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    margin-bottom: 20px;
    font-size: 17px;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #4CAF50, #43A047);
    color: #ffffff; /* Light text on dark button */
    font-size: 16px;
    border: none;
    border-radius: 6px;
    padding: 12px 20px;
    width: 100%;
    cursor: pointer;
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(76, 175, 80, 0.2);
}

/* Result Box */
.result-box {
    background-color: #ffffff;
    color: #2c3e50; /* Ensure dark readable text */
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    margin-top: 30px;
}

/* Cards */
.card {
    background-color: #ffffff;
    color: #2c3e50; /* Ensure text is visible */
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    margin: 15px 0;
    text-align: center;
    transition: transform 0.2s ease-in-out;
}

.card:hover {
    transform: scale(1.02);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    padding: 15px;
    font-size: 14px;
    color: #7f8c8d;
    background-color: transparent;
}

    </style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r".\Employee.csv")
    return data

# Preprocess the data
def preprocess_data(data):
    # Impute missing values
    data.fillna({'EverBenched': 'No', 'ExperienceInCurrentDomain': 0}, inplace=True)
    
    # Label encode binary variables
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])  # Male=1, Female=0
    data['EverBenched'] = label_encoder.fit_transform(data['EverBenched'])  # Yes=1, No=0
    
    # One-hot encode multi-class variables
    data = pd.get_dummies(data, columns=['Education', 'City'], drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['Age', 'JoiningYear', 'ExperienceInCurrentDomain']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    return data

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "salary_prediction_model.pkl"
    if not os.path.exists(model_path):  # Check if the model file exists
        st.error(f"Model file '{model_path}' not found. Please ensure the file exists or retrain the model.")
        st.stop()
    return joblib.load(model_path)

# Main function to run the Streamlit app
def main():
    # Header
    st.markdown('<div class="header">Employee Salary Prediction App</div>', unsafe_allow_html=True)

    # Query Parameters
    query_params = st.query_params
    current_page = query_params.get("page", ["Home"])[0]

    # Navigation Bar
    st.markdown("""
        <div class="navbar">
            <a href="?page=Home" class="{}">🏠 Home</a>
            <a href="?page=Dataset Overview" class="{}">📊 Dataset Overview</a>
            <a href="?page=Model Insights" class="{}">🔍 Model Insights</a>
            <a href="?page=Make a Prediction" class="{}">🎯 Make a Prediction</a>
        </div>
    """.format(
        "active" if current_page == "Home" else "",
        "active" if current_page == "Dataset Overview" else "",
        "active" if current_page == "Model Insights" else "",
        "active" if current_page == "Make a Prediction" else ""
    ), unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.markdown('<p class="sidebar">Navigation</p>', unsafe_allow_html=True)
    sidebar_page = st.sidebar.radio(
        "",
        ["Home", "Dataset Overview", "Model Insights", "Make a Prediction"],
        format_func=lambda x: f"📊 {x}" if x == "Dataset Overview" else f"🔍 {x}" if x == "Model Insights" else f"🎯 {x}" if x == "Make a Prediction" else f"🏠 {x}"
    )

    # Sync sidebar and navbar navigation
    if sidebar_page != current_page:
        st.query_params["page"] = sidebar_page
        current_page = sidebar_page

    # Load dataset and model
    data = load_data()
    data = preprocess_data(data)
    X = data.drop(columns=['PaymentTier'])
    y = data['PaymentTier']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = load_model()

    # Home Page
    if current_page == "Home":
        st.header("Welcome to the Employee Salary Prediction App!")
        st.write("""
            This app uses machine learning to predict the Payment Tier of employees based on their details.
            Navigate through the sidebar or the navigation bar to explore the dataset, view model insights, and make predictions.
        """)

    # Dataset Overview Page
    elif current_page == "Dataset Overview":
        st.subheader("Dataset Overview")
        st.write(data.head())

        # Cards for key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card"><strong>Rows:</strong><br>{}</div>'.format(len(data)), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><strong>Columns:</strong><br>{}</div>'.format(len(data.columns)), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card"><strong>Missing Values:</strong><br>{}</div>'.format(data.isnull().sum().sum()), unsafe_allow_html=True)

        # Expandable sections
        with st.expander("Target Variable Distribution"):
            fig, ax = plt.subplots()
            data['PaymentTier'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_title("Distribution of Payment Tiers")
            st.pyplot(fig)

    # Model Insights Page
    elif current_page == "Model Insights":
        st.subheader("Feature Importance")
        feature_importances = model.feature_importances_
        features = X.columns
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features, feature_importances, color='skyblue')
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Features")
        ax.set_title("Feature Importance from Random Forest")
        st.pyplot(fig)

        st.subheader("Confusion Matrix")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # Make a Prediction Page
    elif current_page == "Make a Prediction":
        st.subheader("Make a Prediction")
        st.write("Enter the details below to predict the Payment Tier:")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            joining_year = st.number_input("Joining Year", min_value=1990, max_value=2023, value=2010)
        with col2:
            education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
            city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
            ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
        experience_in_current_domain = st.number_input("Experience in Current Domain", min_value=0, max_value=50, value=5)

        # Convert inputs to DataFrame
        user_input = pd.DataFrame({
            'Gender': [1 if gender == "Male" else 0],
            'Age': [age],
            'JoiningYear': [joining_year],
            'EverBenched': [1 if ever_benched == "Yes" else 0],
            'ExperienceInCurrentDomain': [experience_in_current_domain],
            'Education': [education],  # Add Education column
            'City': [city]             # Add City column
        })

        # One-hot encode Education and City
        user_input = pd.get_dummies(user_input, columns=['Education', 'City'], drop_first=True)

        # Align user input with training data
        user_input = user_input.reindex(columns=X.columns, fill_value=0)

        # Scale numerical features
        numerical_features = ['Age', 'JoiningYear', 'ExperienceInCurrentDomain']
        scaler = StandardScaler()
        user_input[numerical_features] = scaler.fit_transform(user_input[numerical_features])

        # Make prediction
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                prediction = model.predict(user_input)
            st.markdown('<div class="result-box">The predicted Payment Tier is: <strong>{}</strong></div>'.format(prediction[0]), unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">Developed by Your Name | © 2023</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()