# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:31:03 2023

@author: Vinod
"""
import pandas as pd
import streamlit as st
import pickle

# Load the saved model
loaded_model = pickle.load(open('C:\\Users\\Vinod\\Downloads\\Myocardial_Project3\\bernoulli_naive_bayes_model.pkl', 'rb'))

# Define the selected features
selected_features = ['AGE', 'SEX','RAZRIV', 'K_SH_POST', 'D_AD_ORIT']


# Streamlit Input
st.set_page_config(page_title="Myocardial Infarction Predictor", page_icon="myocard.png")
st.sidebar.header('Predict Lethal Outcome with bernoulli model')

# Initialize an empty dictionary to store input values
input_values = {}

# Collect input values for each selected feature
for feature in selected_features:
    if feature == 'SEX':
        value = st.sidebar.selectbox('Gender', ['female', 'male'], key=feature)
        value = 0 if value == 'female' else 1
    elif feature == 'AGE':
        value = st.sidebar.number_input(f'Enter Age', key=feature, step=1, value=30)
        value = int(value)
    elif feature in ['RAZRIV', 'K_SH_POST', 'D_AD_ORIT']:
        label = feature.replace('_', ' ').title()
        value = st.sidebar.selectbox(f'{label}', ['no', 'yes'], key=feature)
        value = 0 if value == 'no' else 1

    input_values[feature] = value

# Now the input_values dictionary contains user-provided values for each selected feature
#st.sidebar.write("Input values:", input_values)

    

# Convert input values to a DataFrame
input_data = pd.DataFrame([input_values], columns=selected_features)

# Predict the outcome using the loaded model
prediction_class = loaded_model.predict(input_data)[0]

# Define class labels and descriptions
class_labels = {
    0: 'unknown (alive)',
    1: 'cardiogenic shock',
    2: 'pulmonary edema',
    3: 'myocardial rupture',
    4: 'progress of congestive heart failure',
    5: 'thromboembolism',
    6: 'asystole',
    7: 'ventricular fibrillation'
}

# Define class descriptions
class_descriptions = {
    0: "The predicted outcome suggests that the individual is alive and in an unknown condition.",
    1: "The predicted outcome suggests the possibility of cardiogenic shock, a serious condition that requires immediate medical attention.",
    2: "The predicted outcome indicates the potential for pulmonary edema, which involves fluid accumulation in the lungs.",
    3: "The predicted outcome suggests the risk of myocardial rupture, a serious complication of heart attacks.",
    4: "The predicted outcome indicates the potential for progression of congestive heart failure.",
    5: "The predicted outcome suggests the possibility of thromboembolism, which involves blood clot formation.",
    6: "The predicted outcome indicates the possibility of asystole, a type of irregular heartbeat.",
    7: "The predicted outcome suggests the risk of ventricular fibrillation, a serious arrhythmia."
    
}

# Display the predicted class and description
predicted_class_label = class_labels[prediction_class]
st.title("Myocardial Infarction Outcome Predictor")
st.subheader("Predicted Lethal Outcome:")

with st.container():
    st.info(f"Class: {prediction_class} - {predicted_class_label}")
    st.write(class_descriptions[prediction_class])

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Disclaimer: This is a prediction tool and not a substitute for medical advice.")



















