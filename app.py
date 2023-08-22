# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:31:03 2023

@author: Vinod
"""
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer



# Read the CSV file into a DataFrame
df = pd.read_csv('Myocardial infarction complications.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy if needed
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Define the target variable
target_column = 'LET_IS'

# Define the selected features
selected_features = ['AGE', 'SEX', 'RAZRIV', 'K_SH_POST', 'ritm_ecg_p_04', 'SVT_POST', 'endocr_02']

# Split the data into features (X) and target (y)
X = df[selected_features]
y = df[target_column]

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Initialize the GridSearchCV with cross-validation
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform the search on your data
grid_search.fit(X, y)

# Get the best parameters
best_params = grid_search.best_params_

# Initialize the RandomForestClassifier with the best parameters
best_rf_classifier = RandomForestClassifier(**best_params)

# Fit the model on the training data
best_rf_classifier.fit(X, y)

# Streamlit Input
st.set_page_config(page_title="Myocardial Infarction Predictor", page_icon="heart1.png")
st.sidebar.header('Predict Lethal Outcome with Random Forest Classifier')

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

# Create input fields for the selected features in a single column
input_values = {}
for feature in selected_features:
    if feature == 'SEX':
        value = st.sidebar.selectbox('Gender', ['female', 'male'], key=feature)
        value = 0 if value == 'female' else 1
    elif feature in ['zab_leg_03', 'SVT_POST', 'ritm_ecg_p_04', 'n_p_ecg_p_07', 'n_p_ecg_p_10', 'RAZRIV',
                    'endocr_02', 'K_SH_POST', 'FIBR_JELUD', 'n_r_ecg_p_04']:
        if feature == 'RAZRIV':
            label = "Myocardial Rupture"
        elif feature == 'K_SH_POST':
            label = "Cardiogenic Shock After MI"
        elif feature == 'ritm_ecg_p_04':
            label = "ECG Rhythm Abnormality"
        elif feature == 'SVT_POST':
            label = "Supraventricular Tachycardia Post MI"
        elif feature == 'endocr_02':
            label = "Obesity in the Anamnesis (Endocrine Disorder)"
        else:
            label = feature
        value = st.sidebar.selectbox(f'{label}', ['no', 'yes'], key=feature)
        value = 0 if value == 'no' else 1
    elif feature == 'AGE':
        value = st.sidebar.number_input(f'Enter Age', key=feature, step=1, value=30)
        value = int(value)
    input_values[feature] = value

    
    

# Convert input values to a DataFrame
input_data = pd.DataFrame([input_values], columns=selected_features)

# Predict the outcome using the trained model
prediction_class = best_rf_classifier.predict(input_data)[0]

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


















