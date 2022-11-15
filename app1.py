import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("reg_model.pkl", "rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
    print(prediction)
    return prediction



def main():
    st.title("Happy Home")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> House Price Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    CRIM = st.number_input("CRIM",format="%.5f")
    ZN = st.number_input("ZN",format="%.5f")
    INDUS = st.number_input("INDUS",format="%.5f")
    CHAS = st.number_input("CHAS",format="%.5f")
    NOX = st.number_input("NOX",format="%.5f")
    RM = st.number_input("RM",format="%.5f")
    AGE = st.number_input("AGE",format="%.5f")
    DIS = st.number_input("DIS",format="%.5f")
    RAD = st.number_input("RAD",format="%.5f")
    TAX = st.number_input("TAX",format="%.5f")
    PTRATIO = st.number_input("PTRATIO",format="%.5f")
    B = st.number_input("B",format="%.5f")
    LSTAT = st.number_input("LSTAT",format="%.5f")
    result=0
    if st.button("Predict"):
        result=predict_note_authentication(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets have your own house")
        st.text("Built your dreams")

if __name__=='__main__':
    main()
