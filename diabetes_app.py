import pickle
import streamlit as st
import pandas as pd
import numpy as np

model = pickle.load(open("C:/Diabetes_ML_Deployment/diabetes_model.pkl", "rb"))

def welcome():
    return "Welcome All"

def diabetes_prediction(new_data):

    #changing the new_data to numpy array
    new_data_as_numpy_array = np.asarray(new_data)

    #reshape the array as we are predicting for one instance
    new_data_reshaped = new_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(new_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'This individual is not diabetic'
    else:
        return 'This individual is diabetic'

def main():

    #Web app title
    st.title("Diabetes Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Diabetes Prediction ML App </h2> 
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    pregnancies = st.text_input("Number of times Pregnant", "Type Here")
    glucose = st.text_input("Glucose Level ", "Type Here")
    bmi = st.text_input("BMI Value", "Type Here")
    insulin = st.text_input("Insulin Level", "Type Here")
    skinthickness = st.text_input("Skin Thickness Value", "Type Here")
    age = st.text_input("Age of Individual", "Type Here")
    bloodpressure = st.text_input("Blood Pressure Value", "Type Here")

    ## code for prediction
    result = ""
    if st.button("Predict"):
        result = diabetes_prediction([pregnancies, glucose, bmi, insulin, skinthickness, age, bloodpressure])
    st.success(result)    #'The output is {}'.format(result))
    if st.button("About"):
        st.text("Let's Learn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()






#%%
