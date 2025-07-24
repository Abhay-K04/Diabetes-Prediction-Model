# app.py

# -*- coding: utf-8 -*-
"""
Streamlit Web App for Diabetes Prediction
"""

import numpy as np
import pandas as pd
import pickle
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Load or train model
MODEL_FILENAME = 'trained_model.sav'

@st.cache_resource
def train_and_save_model():
    if not os.path.exists(MODEL_FILENAME):
        # Load data
        diabetes_dataset = pd.read_csv('diabetes.csv')

        # Split data
        X = diabetes_dataset.drop(columns='Outcome', axis=1)
        Y = diabetes_dataset['Outcome']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # Train model
        classifier = svm.SVC(kernel='linear')
        classifier.fit(X_train, Y_train)

        # Save model
        pickle.dump(classifier, open(MODEL_FILENAME, 'wb'))
    else:
        print("Model file already exists.")

train_and_save_model()

# Load trained model
loaded_model = pickle.load(open(MODEL_FILENAME, 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Streamlit Web App
def main():
    st.title('Diabetes Prediction Web App')
    st.write("Enter the following values to predict whether a person is diabetic:")

    # Input fields
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Prediction
    diagnosis = ''
    if st.button('Predict Diabetes'):
        try:
            input_list = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            input_list = [float(i) for i in input_list]
            diagnosis = diabetes_prediction(input_list)
        except ValueError:
            diagnosis = "Please enter valid numerical values for all fields."

    st.success(diagnosis)


if __name__ == '__main__':
    main()
