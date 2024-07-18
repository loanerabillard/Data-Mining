import streamlit as st
import pandas as pd
from data_exploration import load_data, data_description
from data_preprocessing import handle_missing_values, normalize_data
from data_visualization import visualize_data
from clustering_prediction import clustering
from learning_evaluation import evaluate_clustering

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 90%;
        padding-left: 5%;
        padding-right: 5%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("Data Mining Project")
    st.sidebar.header("Navigation")
    options = ["Data Exploration", "Data Pre-processing", "Data Visualization", "Clustering", "Evaluation"]
    choice = st.sidebar.radio("Go to", options)
    
    data = load_data()
    if data is not None:
        if choice == "Data Exploration":
            data_description(data)
        elif choice == "Data Pre-processing":
            data = handle_missing_values(data)
            data = normalize_data(data)
        elif choice == "Data Visualization":
            data, label_encoders = handle_missing_values(data)
            data = normalize_data(data)
            visualize_data(data)
        elif choice == "Clustering":
            data, label_encoders = handle_missing_values(data)
            data = normalize_data(data)
            clusters = clustering(data)
        elif choice == "Evaluation":
            data, label_encoders = handle_missing_values(data)
            data = normalize_data(data)
            clusters = clustering(data)
            evaluate_clustering(data, clusters)

if __name__ == "__main__":
    main()
