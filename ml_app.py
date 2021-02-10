import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Page Layout
# Page expands to full width
st.set_page_config(page_title="EOR_Ml App", layout="wide")

