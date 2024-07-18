import streamlit as st
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from clustering_prediction import clustering  
from data_exploration import load_data
from data_preprocessing import handle_missing_values, normalize_data

def evaluate_clustering(data, clusters):
    st.header("Cluster Evaluation")

    numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    if clusters is None:
        st.warning("No clusters available to evaluate.")
        return
    
    silhouette_avg = silhouette_score(data[numerical_columns], clusters)
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
    data_2d = pca.fit_transform(data[numerical_columns])
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(data_2d[:,0], data_2d[:,1], c=clusters, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)


