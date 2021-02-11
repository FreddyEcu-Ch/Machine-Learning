import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Page Layout
# Page expands to full width
st.set_page_config(page_title="EOR_Ml App", layout="wide")

# Insert image
image = Image.open("dt.png")
st.image(image, width=700)

# Write title and additional information
st.title("Data Science for Oil and Gas Engineering")
st.markdown("""
This App consists of implementing an **EOR Screening** by using Machine Learning 
algorithms. It must be mentioned that the dataset used for training and evaluating 
these algorithms have a size of roughly 200 successful EOR Projects (Rows or 
observations) from some countries, as well as 7 reservoir parameters, which are the 
feature or dependent variables. Furthermore, the target variable of this model is a 
categorical variable, which contains 5 EOR Methods (classes) such as the steam injection 
method, CO2 injection method, HC injection method, polymer injection method, and 
combustion in situ method.

* **Author:** [Freddy Carrion](https://www.linkedin.com/in/freddy-carri%C3%B3n-maldonado-b3579b125/)

* **Python Libraries:** scikit-learn, pandas, numpy, streamlit, matplotlib, pandas_profiling
""")

# Sidebar - collects user input features into dataframe
with st.sidebar.header("1. Upload the csv data"):
    upload_file = st.sidebar.file_uploader("Uploado your csv file", type=["csv"])
    st.sidebar.markdown("""
    [Download csv file](https://raw.githubusercontent.com/FreddyEcu-Ch/Machine-Learning/main/DATA%20WORLWIDE%20EOR%20PROJECTSP.csv)
    """)

