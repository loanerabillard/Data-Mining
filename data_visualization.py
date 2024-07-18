import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_exploration import load_data
from data_preprocessing import handle_missing_values, normalize_data

def visualize_data(data):
    st.header("Data Visualization")
    visualization = st.selectbox("Select visualization type:", 
                                 ["Histogram", "Box Plot"])
    
    column = st.selectbox("Select column to visualize:", data.columns)
    
    if visualization == "Histogram":
        st.write(f"Histogram of {column}")
        fig, ax = plt.subplots()
        ax.hist(data[column], bins=30, edgecolor='k')
        st.pyplot(fig)
    
    elif visualization == "Box Plot":
        st.write(f"Box Plot of {column}")
        fig, ax = plt.subplots()
        sns.boxplot(data[column], ax=ax)
        st.pyplot(fig)

