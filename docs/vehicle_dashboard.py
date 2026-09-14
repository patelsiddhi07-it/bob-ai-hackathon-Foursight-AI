import streamlit as st
import pandas as pd

# Sidebar data source selection
data_source = st.sidebar.radio("Data Source", ["file path", "upload CSV"])

if data_source == "file path":
    # State: BEFORE upload (using default local path)
    df = pd.read_csv("data/hums_data.csv")
    
    # Filter the dataframe to only show zero error vehicles
    zero_error_df = df[(df['# Faults'] == 0) & (df['# Warnings'] == 0)]
    
    st.subheader("Zero Error Vehicles")
    st.dataframe(zero_error_df)

elif data_source == "upload CSV":
    # State: AFTER selecting upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Render the full page shown in your image once a file is present
        error_df = pd.read_csv(uploaded_file)
        
        st.subheader("Prioritized Maintenance Work Orders — 10 vehicle(s)")
        st.dataframe(error_df)
    else:
        # Prompt the user if they haven't uploaded anything yet
        st.info("Please upload your error file to view the dashboard.")
