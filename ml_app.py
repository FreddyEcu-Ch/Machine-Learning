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
st.image(image, width=100, use_column_width=True)

# Write title and additional information
st.title("Data Science for Oil and Gas Engineering", )
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
    upload_file = st.sidebar.file_uploader("Upload your csv file", type=["csv"])
    st.sidebar.markdown("""
    [Download csv file](https://raw.githubusercontent.com/FreddyEcu-Ch/Machine-Learning/main/DATA%20WORLWIDE%20EOR%20PROJECTSP.csv)
    """)

# Sidebar - ML Algorithms
with st.sidebar.subheader("2. Select ML Alogrithm"):
    algorithm = st.sidebar.selectbox("Select algorithm", ("KNN", "Decision Tree"))

# Setting parameters
with st.sidebar.subheader("3. Set User Input Parameters"):
    split_size = st.sidebar.slider('Data split ratio (% for training set)', 10, 90, 80)

with st.sidebar.subheader("3.1 Learning parameters"):
    if algorithm == "KNN":
        parameter_k_neighbors = st.sidebar.slider("Number of K neighbors", 1, 30, 2)

    else:
        parameter_decision_tree = st.sidebar.slider("Number of max depth", 1,10, 3)

with st.sidebar.subheader("3.2 Reservoir Parameters"):
    Porosity = st.sidebar.slider("Porosity (%)", 2, 30)
    Permeability = st.sidebar.slider("Permeability (md)", 8, 5000)
    Depth = st.sidebar.slider("Depth (ft)", 1000, 10000, 1200)
    Gravity = st.sidebar.slider("API Gravity", 5, 40, 8)
    Viscosity = st.sidebar.slider("Oil Viscosity (cp)", 10, 500000, 490058)
    Temperature = st.sidebar.slider("Reservoir Temperature (F)", 50, 300)
    Oil_saturation = st.sidebar.slider("Oil Saturation (%)", 10, 80, 35)

# Exploratory Data Analysis (EDA)
st.write('---')
if st.button('Exploratory Data Analysis (EDA)'):
    st.header('**Exploratory Data Analysis (EDA)**')
    st.write('---')
    if upload_file is not None:
        @st.cache
        def load_csv():
            data = pd.read_csv(upload_file)
            return data
        df = load_csv()
        report = ProfileReport(df, explorative=True)
        st.markdown('**Input Dataframe**')
        st.write(df)
        st.write('---')
        st.markdown('**Pandas Profiling Report**')
        st_profile_report(report)
