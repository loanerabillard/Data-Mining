import streamlit as st
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from clustering_prediction import clustering  # Import clustering function from clustering.py
from data_exploration import load_data
from data_preprocessing import handle_missing_values, normalize_data

def evaluate_clustering(data, clusters):
    st.header("Cluster Evaluation")
    
    if clusters is None:
        st.warning("No clusters available to evaluate.")
        return
    
    silhouette_avg = silhouette_score(data, clusters)
    st.write(f"Silhouette Score: {silhouette_avg}")
    
    # Calculate cluster statistics
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    cluster_stats = pd.DataFrame({
        'Cluster': unique_clusters,
        'Count': counts
    })
    
    st.write("Cluster Statistics:")
    st.write(cluster_stats)
    
    # If K-Means clusters, also show cluster centers
    if isinstance(clusters, np.ndarray):
        if len(np.unique(clusters)) > 1:  # Ensure there's more than one cluster for K-Means
            try:
                kmeans = KMeans(n_clusters=len(np.unique(clusters)))
                kmeans.fit(data)
                cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns)
                st.write("Cluster Centers (K-Means):")
                st.write(cluster_centers)
            except ValueError:
                pass
    
    # Visualize clusters using PCA
    pca = PCA(2)
    data_2d = pca.fit_transform(data)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(data_2d[:,0], data_2d[:,1], c=clusters, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)

def main():
    st.title("Data Mining Project")
    st.header("Cluster Evaluation")
    data = load_data()
    if data is not None:
        data = handle_missing_values(data)
        data = normalize_data(data)
        clusters = clustering(data)  # Capture clusters from clustering.py
        evaluate_clustering(data, clusters)  # Pass data and clusters for evaluation

if __name__ == "__main__":
    main()
