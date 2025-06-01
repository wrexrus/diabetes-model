import numpy as np
import pickle
import streamlit as st

loaded_model_svm = pickle.load(open('D:/Diabetes-pred-model/deploy_svm_dataset1/trained_model_svm_dataset1.sav','rb'))

#creating a function for prediction

def diabetes_prediction(input_data):

        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)

        prediction = loaded_model_svm.predict(input_data_reshaped)
        print(prediction)

        if(prediction[0] == 0):
          return "The person is not Diabetic"
        else:
          return "The person is Diabetic"

def main():
    
    
    #title
    st.title('Diabetes Prediction Model')
    
    
    #input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Pedigree Function')
    Age = st.text_input('Age')
    
    #code for Prediction
    diagnosis = ''
    
    #creating a button
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)


if __name__ == '__main__':
    main()    