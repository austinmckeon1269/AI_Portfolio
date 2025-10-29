
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv('data/churn_data.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train model
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
model = LogisticRegression().fit(X_scaled, y)

# App UI
st.title('Customer Churn Prediction')
st.sidebar.header('Input Customer Data')

def user_input_features():
    data = {}
    for col in X.columns:
        data[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].median()))
    return pd.DataFrame([data])

input_df = user_input_features()
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader('Prediction')
st.write('Churn' if prediction[0] == 1 else 'No Churn')
st.subheader('Prediction Probability')
st.write(prediction_proba)
