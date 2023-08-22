# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:12:39 2023

@author: vinod
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st

#page_bg_img = '''
#<style>
#.stApp {
#height: cover;
#width: cover;
#background-size: cover;
#background-color: #FF0000;
#}
#</style>
#'''
#st.markdown(page_bg_img, unsafe_allow_html=True)

# loading the saved model
loaded_model = pickle.load(open('C:\\Users\\Vinod\\Downloads\\Myocardial_Project3\\rf_model.pkl','rb'))


# creating a function for Prediction


# Placeholder function to simulate loaded model
class LoadedModel:
    def predict(self, input_data):
        # Placeholder logic
        return np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])

loaded_model = LoadedModel()

def heartattack_prediction(input_data):
    input_np = np.asarray(input_data)
    input_data_reshaped = input_np.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction == 0:
        return 'Unknown (alive)'
    elif prediction == 1:
        return 'Cardiogenic shock'
    elif prediction == 2:
        return 'Pulmonary edema'
    elif prediction == 3:
        return 'Myocardial Rupture'
    elif prediction == 4:
        return 'Progress of congestive heart failure'
    elif prediction == 5:
        return 'Thromboembolism'
    elif prediction == 6:
        return 'Asystole'
    elif prediction == 7:
        return 'Ventricular fibrillation'

def main():
    st.sidebar.title("Input the Myocardial Infarction Factors:")
    
    Age = st.sidebar.number_input("Enter an Age:", value=0.0)
    Gender = st.sidebar.selectbox("Select the Gender of the person: 0-Female, 1-Male", ("0", "1"))
    # Add more input fields
    
    # Collect other input fields
    
    input_data = [Age, Gender, ...]  # Collect all input fields into a list
    
    try:
        predict_button = st.sidebar.button("Predict")  # Add a prediction button
    except Exception as e:
        predict_button = False
    
    if predict_button:
        result = heartattack_prediction(input_data)
        st.write("Prediction:", result)

if __name__ == "__main__":
    main()



  
    
    