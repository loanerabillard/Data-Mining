import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from data_exploration import load_data



def handle_missing_values(data):
    
    numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    

    st.subheader("Handle Missing Values")
    method = st.selectbox("Choose a method to handle missing values", 
                            ['Delete rows', 'Delete columns', 'Replace with mean', 'Replace with median', 
                            'Replace with mode', 'KNN imputation'])
    
    missing_value_method_cat = st.selectbox(
        "Choose a method to handle missing values in categorical data:",
        ["Delete rows/columns", "Replace with most frequent", "Replace with constant"]
    )

    col1, col2 = st.columns(2)
    

    with(col1):
        if method == 'Delete rows':
            data.dropna(subset=numerical_columns)  
        elif method == 'Delete columns':
            data.dropna(subset=numerical_columns)
        elif method == 'Replace with mean':
            imputer = SimpleImputer(strategy='mean')
            data[numerical_columns] = imputer.fit_transform(data[numerical_columns])  
        elif method == 'Replace with median':
            imputer = SimpleImputer(strategy='median')
            data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
        elif method == 'Replace with mode':
            imputer = SimpleImputer(strategy='most_frequent')
            data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
        elif method == 'KNN imputation':
            imputer = KNNImputer(n_neighbors=5)
            data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

        if missing_value_method_cat == "Delete rows/columns":
            data = data.dropna(subset=categorical_columns)
        elif missing_value_method_cat == "Replace with most frequent":
            imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_columns] = imputer.fit_transform(data[categorical_columns])
        elif missing_value_method_cat == "Replace with constant":
            constant_value = st.text_input("Enter the constant value for replacement:")
            imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
            data[categorical_columns] = imputer.fit_transform(data[categorical_columns])


        st.dataframe(data)
    
    apply_button = st.button("Apply This Method to Official DataFrame")

    if apply_button:
        st.success("Method applied successfully to the DataFrame.")
        st.dataframe(data)
        return data
        
    with(col2):
        if data is not None:
            st.write("Number of missing values per column:")
            st.write(data.isnull().sum())
    
    return data
    
    
    






def normalize_data(data):
    st.header("Data Normalization")
    method = st.selectbox("Select a normalization method:", 
                          ["Min-Max Normalization", "Z-score Standardization"])
    
    
    numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    data_numeric = data[numerical_columns]

    if method == "Min-Max Normalization":
        scaler = MinMaxScaler()
    elif method == "Z-score Standardization":
        scaler = StandardScaler()
    
    data_normalized = pd.DataFrame(scaler.fit_transform(data_numeric), columns=data_numeric.columns)
    
    for col in data.columns:
        if col not in numerical_columns:
            data_normalized[col] = data[col]
    
    st.write("Data after normalization:")
    st.dataframe(data_normalized.head())
    return data_normalized


