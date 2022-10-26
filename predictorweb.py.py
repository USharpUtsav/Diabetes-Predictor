# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:03:34 2022

@author: ASUS
"""

import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open('C:/Users/ASUS/Downloads/Deployment/trained_model.sav','rb'))

#function for web page for prediction
def diabetes(input_data):
    input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

def main():
    #Tile for webpage
    st.title('Diabetes Predictor||')
    #input data from the user for the following features
    Pregnancies=st.text_input('No of Pregnancies')
    Gulcose=st.text_input('Gulcose Level')
    BloodPressure=st.text_input('BP level')
    SkinThickness=st.text_input('Skin Thickness Value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI Level')
    DiabetesPredictorFunction=st.text_input('Diabetes Pedigree Function Value')
    Age=st.text_input('Age')
    
    #code for predicting in web
    diagnosis=''
    
    #button for prediction
    if st.button('Test Result'):
        diagnosis=diabetes([Pregnancies,Gulcose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPredictorFunction,Age])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
    
    
    
    
    
    
    

    