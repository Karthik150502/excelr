import pandas as pd
import pickle
import streamlit as st

log_model = pickle.load(open("log_reg_assignment.pkl", "rb"))
std_scaler = pickle.load(open("std_scaler_log_reg_assingment.pkl", "rb"))

st.title("Finding Diabetes probability using Logistic Regression")

def get_user_inputs():

  pregnancies = st.sidebar.slider("Enter the number of pregnancies",0, 50)
  
  glucose = st.sidebar.number_input("Enter the glucose value")
  skinThickness = st.sidebar.number_input("Enter the skinThickness value")
  insulin = st.sidebar.number_input("Enter the Insulin value")
  diabetesPedigreeFunction = st.sidebar.number_input("Enter the Diabetes Pedigree Function value")
  age = st.sidebar.slider("Enter the Age", 0, 150)
  return {
      "Pregnancies":round(pregnancies),
      "Glucose":glucose,
      "SkinThickness":skinThickness,
      "Insulin":insulin,
      "DiabetesPedigreeFunction":diabetesPedigreeFunction,
      "Age":round(age)
  }



def build_features(values, scale_cols):
  features = pd.DataFrame(values, index=[0])
  features[scale_cols]= std_scaler.transform(features[scale_cols])
  return features

user_inputs = get_user_inputs()
df = build_features(user_inputs, ["Glucose", "SkinThickness", "Insulin"])
pred = log_model.predict(df)
pred_prob = log_model.predict_proba(df)
button = st.button('Predict')
if button is True:
    st.subheader('Predicted output')
    st.write('Diabetes Positive' if pred_prob[0][1]>=0.5 else 'Diabetes Negative')
    st.subheader('Predicted_probabilities')
    st.write(pred_prob)