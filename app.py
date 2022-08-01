import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Holt-Winters')

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLMSEX = st.sidebar.selectbox('Gender',('1','0'))
    CLMAGE = st.sidebar.number_input("Insert the No: of Days")
    data = {'CLMSEX':CLMSEX,
            'CLMAGE':CLMAGE}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('Gold Prediction.py ', 'rb'))

prediction = loaded_model.fitted_model(data)
prediction_proba = loaded_model.test_predictions(data)


st.subheader('Prediction Probability')
st.write(prediction_proba)
