import streamlit as st
import pandas as pd
import io


def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        sep = st.text_input("Enter the separator used in the file (e.g., ',' for comma, ';' for semicolon):", value=",")
        header_row = st.number_input("Enter the header row number (0 if no header):", min_value=0, value=0)
        try:
            data = pd.read_csv(uploaded_file, sep=sep, header=header_row)
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def data_description(data):
    st.write("First few rows of the data:")
    st.dataframe(data.head())
    st.write("Last few rows of the data:")
    st.dataframe(data.tail())
    st.write("Data statistical summary:")
    st.write(data.describe())
    # st.write("Data info:")
    # st.write(data.info())
    st.write("Data info:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    
    # Parsing the info string to create a DataFrame
    info_list = []
    lines = s.strip().split('\n')
    for line in lines:
        if line.startswith(' '):
            continue
        col_info = line.split(':')
        col_name = col_info[0]
        col_value = ':'.join(col_info[1:]).strip()
        info_list.append([col_name, col_value])
    
    df = pd.DataFrame(info_list, columns=['Column', 'Info'])
    st.dataframe(df)

def main():
    st.title("Data Mining Project")
    st.header("Initial Data Exploration")
    data = load_data()
    if data is not None:
        data_description(data)

if __name__ == "__main__":
    main()
