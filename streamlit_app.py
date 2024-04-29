import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from locationCOMID_input import get_gages_by_comid
import model

st.title("StreamSage 2.0")
st.write("*An interactive tool to predict streamflow in ungaged locations across California.*")
st.write(
    """
    ## How it Works
    StreamSage 2.0 leverages USGS historical streamflow data from existing gages across California to predict streamflows at locations without gages. 
    Enter the desired Gage ID to get predicted stream flow. More details about our model please refer to our [Github repository](https://github.com/ChristianaKang11/DSSxTNC.git)
    """,
    unsafe_allow_html=True
)

with st.sidebar:
     #Location(Gage ID) Input
    st.subheader("Configure Prediction")
    gage_id = st.text_input("Enter Gage ID", placeholder="Type here...")
    if gage_id:
        gage_id = int(gage_id)
    # st.write("or")
    # COMID= st.text_input("Enter COMID", placeholder="Type here...")

    
    # Button to perform calculation
    calculate_button = st.button("Calculate Flow", help="Click to calculate the predicted streamflow")
    if calculate_button:
        st.success("Calculating streamflow for the selected GageID...")

if calculate_button:
    prediction = model.predictedflow(gage_id, window_start, window_end)
    st.write(prediction)

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
# #if COMID:
#      gage_ids=get_gages_by_comid(data, COMID) 
