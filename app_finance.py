import streamlit as st 
import pandas as pd
from preprocessing import feature,split
from back import result

st.title("Mon application financière")

df = pd.read_csv("ibm.us.txt")

df = feature(df)

X_train,X_test,y_train,y_test = split(df)

output,fig = result(X_test)

st.bokeh_chart(fig)

st.write("Buy & Hold Return [%]",output["Buy & Hold Return [%]"])