import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
    
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    if file is not None:
        Sep = st.text_input("Enter the separator used in the file (e.g., ',' for comma, ';' for semicolon):", value=",")
        Header_RowNumber = st.number_input("Enter the header row number (0 if no header):", min_value=0, value=0)
        try:
            data = pd.read_csv(file, sep=Sep, header=Header_RowNumber)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            
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

        numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        

        st.subheader("Handle Missing Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            method = st.selectbox("Choose a method to handle missing values", 
                                    ['Delete rows', 'Delete columns', 'Replace with mean', 'Replace with median', 
                                    'Replace with mode', 'KNN imputation'])
        
        with col2:
            missing_value_method_cat = st.selectbox(
                "Choose a method to handle missing values in categorical data:",
                ["Delete rows/columns", "Replace with most frequent", "Replace with constant"]
            )

        
        
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

        with(col2):
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
            
        st.header("Data Normalization")
        col1, col2 = st.columns(2)
        
        with col1:
            norm_method = st.selectbox("Choose a normalization method", 
                                       ['None', 'Min-Max Scaling', 'Z-score Standardization'])
        
        with col2:
            if norm_method == 'Min-Max Scaling':
                scaler = MinMaxScaler()
                data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
            elif norm_method == 'Z-score Standardization':
                scaler = StandardScaler()
                data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        
        st.header("Data Visualization")
        visualization = st.selectbox("Select visualization type:", 
                                    ["Histogram", "Box Plot"])
        
        column = st.selectbox("Select column to visualize:", data.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if visualization == "Histogram":
                st.write(f"Histogram of {column}")
                fig, ax = plt.subplots()
                ax.hist(data[column], bins=30, edgecolor='k')
                st.pyplot(fig)
        
        with col2:
            if visualization == "Box Plot":
                st.write(f"Box Plot of {column}")
                fig, ax = plt.subplots()
                sns.boxplot(data[column], ax=ax)
                st.pyplot(fig)
        
        st.header("Clustering")
        col1, col2 = st.columns(2)

        with col1:
            algorithm = st.selectbox("Select clustering algorithm:", ["K-Means", "DBSCAN"])
        
        with col2:
            if algorithm == "K-Means":
                n_clusters = st.number_input("Enter the number of clusters:", min_value=2, value=3)
                kmeans = KMeans(n_clusters=n_clusters)
                clusters = kmeans.fit_predict(data[numerical_columns])
                data['Cluster'] = clusters
                st.write("Cluster centers:")
                st.write(kmeans.cluster_centers_)
            
            elif algorithm == "DBSCAN":
                eps = st.slider("Select eps value:", min_value=0.1, max_value=10.0, value=0.5)
                min_samples = st.number_input("Enter the minimum number of samples per cluster:", min_value=1, value=5)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(data[numerical_columns])
                data['Cluster'] = clusters
        
        with col1:
            st.write("Clusters:")
            st.write(clusters)
        
        pca = PCA(3)
        data_3d = pca.fit_transform(data[numerical_columns])
        
        st.header("3D Cluster Visualization")
        fig = px.scatter_3d(
            x=data_3d[:, 0],
            y=data_3d[:, 1],
            z=data_3d[:, 2],
            color=clusters,
            title="3D Scatter plot of Clusters",
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'z': 'PCA Component 3'}
        )
        st.plotly_chart(fig)
        
        st.header("Cluster Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col2:
            if clusters is None:
                st.warning("No clusters available to evaluate.")
            
            silhouette_avg = silhouette_score(data[numerical_columns], clusters)
            st.write(f"Silhouette Score: {silhouette_avg}")
        
        with col1:
            unique_clusters, counts = np.unique(clusters, return_counts=True)
            cluster_stats = pd.DataFrame({
                'Cluster': unique_clusters,
                'Count': counts
            })
            
            st.write("Cluster Statistics:")
            st.write(cluster_stats)
            
            if isinstance(clusters, np.ndarray):
                if len(np.unique(clusters)) > 1:
                    try:
                        kmeans = KMeans(n_clusters=len(np.unique(clusters)))
                        kmeans.fit(data)
                        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns)
                        st.write("Cluster Centers (K-Means):")
                        st.write(cluster_centers)
                    except ValueError:
                        pass
        
        pca = PCA(2)
        data_2d = pca.fit_transform(data[numerical_columns])
        
        fig, ax = plt.subplots()
        scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)

        st.header("Prediction")
        
        col1, col2 = st.columns(2)

        numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        with col1:
            task = st.selectbox("Select task type:", ["Regression", "Classification"])
        
        with col2:
            if task == "Regression":
                target_column = st.selectbox("Select the target column:", data[numerical_columns].columns)
            elif task == "Classification":
                target_column = st.selectbox("Select the target column:", data[categorical_columns].columns)
        
        features = data.drop(columns=[target_column]).columns.tolist()
        
        # Encode categorical columns
        for col in categorical_columns:
            if col in features:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])

        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider("Select the test size (fraction):", min_value=0.1, max_value=0.5, value=0.2)
        
        with col2:
            random_state = st.number_input("Enter the random state for splitting the data:", min_value=0, value=42)
        
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_column], test_size=test_size, random_state=random_state)
        
        if task == "Regression":
            model_type = st.selectbox("Select regression model:", ["Linear Regression", "Random Forest Regressor"])
            
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Random Forest Regressor":
                n_estimators = st.number_input("Enter the number of estimators:", min_value=10, value=100)
                model = RandomForestRegressor(n_estimators=n_estimators)
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            st.write("Evaluation Metrics:")
            st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, predictions)}")
            st.write(f"R-squared (R2): {r2_score(y_test, predictions)}")
        
        elif task == "Classification":
            model_type = st.selectbox("Select classification model:", ["Logistic Regression", "Random Forest Classifier"])
            
            if model_type == "Logistic Regression":
                model = LogisticRegression()
            elif model_type == "Random Forest Classifier":
                n_estimators = st.number_input("Enter the number of estimators:", min_value=10, value=100)
                model = RandomForestClassifier(n_estimators=n_estimators)
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            st.write("Evaluation Metrics:")
            st.write(f"Accuracy: {accuracy_score(y_test, predictions)}")
            st.write(f"Precision: {precision_score(y_test, predictions, average='weighted')}")
            st.write(f"Recall: {recall_score(y_test, predictions, average='weighted')}")
            st.write(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")

if __name__ == "__main__":
    main()
