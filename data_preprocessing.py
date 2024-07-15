import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from data_exploration import load_data

def handle_missing_values(data):
    st.header("Handling Missing Values")
    method = st.selectbox("Select a method to handle missing values:", 
                          ["Drop rows", "Drop columns", "Mean", "Median", "Mode", "KNN Imputation"])
    
    # Encode non-numeric columns
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = data[column].astype(str)  # Ensure all data are strings
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    if method == "Drop rows":
        data = data.dropna()
    elif method == "Drop columns":
        data = data.dropna(axis=1)
    elif method in ["Mean", "Median", "Mode"]:
        strategy = method.lower()
        imputer = SimpleImputer(strategy=strategy)
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    elif method == "KNN Imputation":
        imputer = KNNImputer()
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    st.write("Data after handling missing values:")
    st.dataframe(data.head())
    return data, label_encoders

def normalize_data(data):
    st.header("Data Normalization")
    method = st.selectbox("Select a normalization method:", 
                          ["Min-Max Normalization", "Z-score Standardization"])
    
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    data_numeric = data[numeric_columns]

    if method == "Min-Max Normalization":
        scaler = MinMaxScaler()
    elif method == "Z-score Standardization":
        scaler = StandardScaler()
    
    data_normalized = pd.DataFrame(scaler.fit_transform(data_numeric), columns=data_numeric.columns)
    
    for col in data.columns:
        if col not in numeric_columns:
            data_normalized[col] = data[col]
    
    st.write("Data after normalization:")
    st.dataframe(data_normalized.head())
    return data_normalized

def main():
    st.title("Data Mining Project")
    st.header("Data Pre-processing and Cleaning")
    data = load_data()
    if data is not None:
        data, label_encoders = handle_missing_values(data)
        data = normalize_data(data)

if __name__ == "__main__":
    main()
