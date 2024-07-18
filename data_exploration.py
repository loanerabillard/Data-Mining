import streamlit as st
import pandas as pd



def load_data():
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    if file is not None:
        Sep = st.text_input("Enter the separator used in the file (e.g., ',' for comma, ';' for semicolon):", value=",")
        Header_RowNumber = st.number_input("Enter the header row number (0 if no header):", min_value=0, value=0)
        try:
            data = pd.read_csv(file, sep=Sep, header=Header_RowNumber)
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

## Rajouter la possibilité de rajouter directement les header 

def data_description(data):
    st.write("Data preview:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("10 first rows of the data: ")
        st.dataframe(data.head(10))

        st.write("Data statistical :")
        st.write(data.describe())
        
    with col2:
        st.write("10 last rows of the data: ")
        st.dataframe(data.tail(10))

        st.write("Number of missing values per column:")
        st.write(data.isnull().sum())
