# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:14:13 2023

@author: Vinod
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:\\Users\\Vinod\\Downloads\\Myocardial_Project3\\rf_model.pkl', 'rb'))

# Define function for prediction
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
    st.title('Classification of Myocardial Infarction')

    # Sidebar inputs
    st.sidebar.title("Input the Myocardial Infarction Factors:")
    Age = st.sidebar.number_input("Enter Age:", value=0)
    Gender = st.sidebar.selectbox("Select Gender: 0-Female, 1-Male", ("0", "1"))
    # ... other inputs ...

    # Prediction button
    if st.sidebar.button('Predict Classification'):
        result = heartattack_prediction([Age, Gender, RAZRIV, D_AD_ORIT, S_AD_ORIT, SIM_GIPERT, ROE, K_SH_POST, TIME_B_S, R_AB_3_n, ant_im, AST_BLOOD, IBS_POST, nr07])
        st.sidebar.write('Classification:', result)

    # File uploader
    uploaded_resume = st.file_uploader("Upload CSV file:", type=["csv"])

    if uploaded_resume is not None:
        df = pd.read_csv(uploaded_resume, index_col=0)
        predict = loaded_model.predict(df)
        df['LET_IS'] = predict

        st.write(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV file", data=csv, file_name='Classification.csv', mime='csv')

if __name__ == '__main__':
    main()
