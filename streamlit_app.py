import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from locationCOMID_input import get_gages_by_comid

st.header("StreamSage 2.0")
st.write("*An interactive tool to predict streamflow in ungaged locations across California.*")
st.write(
    """
    ## How it Works
    StreamSage 2.0 leverages USGS historical streamflow data from existing gages across California to predict streamflows at locations without gages. 
    Enter the desired location/COMID to predict streamflow and select reference gages based on the suggestions provided.
    """
)
with st.sidebar:
    #Location input
    st.subheader("Configure Prediction")
    location_input = st.text_input("Enter location", placeholder="Type here...")
    st.write("or")
    COMID= st.text_input("Enter COMID", placeholder="Type here...")
    # Date range selection
    st.subheader("Select Date Range")
    window_start = st.date_input("Start Date", datetime(2010, 1, 1))
    window_end = st.date_input("End Date", datetime(2022, 12, 31))
    # Gage selection
    st.subheader("Gage Selection")
    st.write("Select reference gages that will be used for predicting the streamflow.")

    # Dummy data for gage selection (This would be dynamic based on actual data)
    gage_options = ['Gage 1', 'Gage 2', 'Gage 3', 'Gage 4', 'Gage 5'] #Replace with output of dynamic windowing
    selected_gages = st.multiselect("Select Gages", options=gage_options, default=gage_options[:20])
    
    # Button to perform calculation
    calculate_button = st.button("Calculate Flow", help="Click to calculate the predicted streamflow")
    if calculate_button:
        st.success("Calculating streamflow for the selected location and gages...")

st.write("---")
st.write("## About StreamSage 2.0")
st.markdown(
    """
    StreamSage 2.0 is a part of a larger initiative of The Nature Conservancy to improve water resource management in California. 
    This streamlit app is developed by [Data Science Society at Berkeley](https://dssberkeley.com/). 
    This tool aims to help researchers, policymakers, and the general public understand potential water 
    availability and make informed decisions regarding water use in ungaged regions.
    """,
    unsafe_allow_html=True
)

# comment for now
# @st.cache
# def load_data():
#     data = pd.read_csv("path.csv")
#     return data
# data = load_data()
# gage_ids=get_gages_by_comid(data, COMID)