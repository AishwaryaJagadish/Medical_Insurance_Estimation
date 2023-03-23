# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 01:50:00 2023

@author: Aishwarya
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('F:/Projects 2022/PYTHON/Medical Insurance Prediction/insurance_model.sav', 'rb'))


# creating a function for Prediction

def insurance_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return prediction[0]
  
      
def main():
    
    
    # giving a title
    st.title('Medical Insurance Estimation')
    
    
    # getting the input data from the user
    
    
    age = st.text_input('Age')
    sex = st.text_input('Sex => Male-0 Female-1')
    bmi = st.text_input('BMI')
    children= st.text_input('Children')
    smoker = st.text_input('Smoker => Yes-0 No-1')
    region = st.text_input('Region => Southeast-0 Southwest-1 Northeast-2 Northwest-3')
    
    
    # code for Prediction
    insurance = ''
    
    # creating a button for Prediction
    
    if st.button('Medical Insurance Cost Prediction'):
        insurance = insurance_prediction([float(age),float(sex),float(bmi),float(children),float(smoker),float(region)])
        st.text('The Medical Insurance is at a cost of USD ')
     
        
    st.success(insurance)
    
        
if __name__ == '__main__':
    main()