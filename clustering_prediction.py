import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from data_exploration import load_data
from data_preprocessing import handle_missing_values, normalize_data

def clustering(data):
    st.header("Clustering")
    algorithm = st.selectbox("Select clustering algorithm:", ["K-Means", "DBSCAN"])
    
    if algorithm == "K-Means":
        n_clusters = st.number_input("Enter the number of clusters:", min_value=2, value=3)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(data)
        st.write("Cluster centers:")
        st.write(kmeans.cluster_centers_)
        
    elif algorithm == "DBSCAN":
        eps = st.slider("Select eps value:", min_value=0.1, max_value=10.0, value=0.5)
        min_samples = st.number_input("Enter the minimum number of samples per cluster:", min_value=1, value=5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data)
    
    st.write("Clusters:")
    st.write(clusters)
    
    pca = PCA(2)
    data_2d = pca.fit_transform(data)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(data_2d[:,0], data_2d[:,1], c=clusters, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)
    
    return clusters

def main():
    st.title("Data Mining Project")
    st.header("Clustering")
    data = load_data()
    if data is not None:
        data = handle_missing_values(data)
        data = normalize_data(data)
        clustering(data)

if __name__ == "__main__":
    main()
