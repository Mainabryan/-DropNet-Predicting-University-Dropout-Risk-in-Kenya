import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

# Set page configuration
st.set_page_config(page_title="Kenya University Dropout Risk Predictor", layout="wide")

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        # Encode categorical variable
        le = LabelEncoder()
        data['financial_stress_level'] = le.fit_transform(data['financial_stress_level'])
        return data, le
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'kenyan_students_funding_2024_2025.csv' is in the same directory.")
        return None, None

# Function to train the logistic regression model
def train_model(data):
    features = ['household_income', 'got_helb', 'helb_amount', 'scholarship_amount', 
                'program_cost_per_year', 'financial_stress_level']
    X = data[features]
    y = data['likely_to_dropout']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    return model, scaler, features

# Function to create visualizations
def create_visualizations(data):
    st.subheader("Visual Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot 1: Distribution of Financial Stress Level
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='financial_stress_level', data=data, palette='viridis')
        ax.set_title("Distribution of Financial Stress Levels")
        ax.set_xlabel("Financial Stress Level")
        ax.set_ylabel("Count")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Low', 'Medium', 'High'])
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Plot 2: Dropout Risk by HELB Funding
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='got_helb', y='likely_to_dropout', data=data, palette='magma')
        ax.set_title("Dropout Risk by HELB Funding")
        ax.set_xlabel("Received HELB (0 = No, 1 = Yes)")
        ax.set_ylabel("Average Dropout Risk")
        plt.tight_layout()
        st.pyplot(fig)

# Function to get user inputs
def get_user_inputs():
    st.subheader("Student Financial Inputs")
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            household_income = st.slider("Monthly Household Income (KES)", 
                                      min_value=3000, max_value=150000, value=35000, step=1000)
            program_cost = st.slider("Annual Program Cost (KES)", 
                                   min_value=60000, max_value=250000, value=100000, step=5000)
            financial_stress = st.selectbox("Financial Stress Level", 
                                          options=['Low', 'Medium', 'High'])
        
        with col2:
            got_helb = st.checkbox("Received HELB Loan")
            helb_amount = st.slider("HELB Loan Amount (KES)", 
                                  min_value=0, max_value=60000, value=0, step=5000) if got_helb else 0
            scholarship_amount = st.slider("Scholarship Amount (KES)", 
                                        min_value=0, max_value=100000, value=0, step=5000)
        
        submitted = st.form_submit_button("Predict Dropout Risk")
        return (submitted, household_income, got_helb, helb_amount, 
                scholarship_amount, program_cost, financial_stress)

# Function to predict dropout risk
def predict_dropout(model, scaler, le, features, user_inputs):
    household_income, got_helb, helb_amount, scholarship_amount, program_cost, financial_stress = user_inputs
    stress_encoded = le.transform([financial_stress])[0]
    
    input_data = np.array([[household_income, got_helb, helb_amount, 
                           scholarship_amount, program_cost, stress_encoded]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict_proba(input_scaled)[0][1]  # Probability of dropout (class 1)
    return prediction

# Main app
def main():
    st.title("Kenya University Dropout Risk Predictor")
    st.markdown("**Navigate through the sections below to explore the data and predict dropout risk.**")
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    st.sidebar.markdown("Use the sections below to move through the app:")
    section = st.sidebar.radio("Go to:", ["Overview", "Visual Insights", "Predict Dropout Risk"])
    
    # Load data
    data, le = load_and_preprocess_data('kenyan_students_funding_2024_2025.csv')
    if data is None:
        return
    
    # Train model
    model, scaler, features = train_model(data)
    
    # Overview Section
    if section == "Overview":
        st.header("Overview")
        st.markdown("""
        This app predicts the risk of university dropout for Kenyan students based on financial factors, 
        particularly in light of the 2024-2025 HELB funding cuts. The model uses logistic regression to 
        analyze factors like household income, HELB loans, scholarships, and financial stress levels.
        
        **Why this matters**: With a reported Sh49.8 billion shortfall in HELB funding, many students face 
        increased financial stress, potentially leading to higher dropout rates. This app helps identify at-risk students.
        
        Navigate to **Visual Insights** to explore the data or **Predict Dropout Risk** to input student details.
        """)
    
    # Visual Insights Section
    elif section == "Visual Insights":
        create_visualizations(data)
        st.markdown("---")
        st.markdown("**Insight**: High financial stress and lack of HELB funding significantly increase dropout risk.")
    
    # Predict Dropout Risk Section
    elif section == "Predict Dropout Risk":
        st.header("Predict Dropout Risk")
        submitted, *user_inputs = get_user_inputs()
        
        if submitted:
            st.markdown("---")
            st.subheader("Prediction Output")
            dropout_prob = predict_dropout(model, scaler, le, features, user_inputs)
            
            # Color-coded result
            color = "green" if dropout_prob < 0.5 else "red"
            st.markdown(f"<h3 style='color:{color}'>Dropout Risk: {dropout_prob:.2%}</h3>", unsafe_allow_html=True)
            
            if dropout_prob < 0.5:
                st.success("Low risk of dropping out. Financial support seems sufficient.")
            else:
                st.error("High risk of dropping out. Consider additional financial support or interventions.")
    
    st.markdown("---")
    st.markdown("Developed by xAI | Data Source: Synthetic Kenyan Student Funding Dataset 2024-2025")

if __name__ == "__main__":
    main()