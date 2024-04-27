import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from copy_of_sooyeon_manual_similarity_measure_expert_rescale_rfe import *
import ast

accesstoken_mark ='pk.eyJ1IjoibXJraWViYXJyIiwiYSI6ImNsb2Rxbm1tZzA0aHUyeHBmMGRhY2NtZ3UifQ.sfAJYCoOIo6s4Zm6-PrmJg',

# Function to load data from CSV with proper caching
@st.cache_data
def load_data(csv_name):
    data = pd.read_csv(csv_name)
    return data

#meowy is dynamic window output from copy df -- this is default for now
dynamic_output_df = meowy
geometries_for_gages_df = merged_gdf

# Sidebar for Dynamic Window input
st.sidebar.header("Dynamic Window Settings")
siteid = st.sidebar.selectbox('Select Target Gage:', pd.unique(geometries_for_gages_df['siteid']))
window_start = st.sidebar.date_input("Window Start Date", datetime.now() - timedelta(days=30))
window_end = st.sidebar.date_input("Window End Date", datetime.now())
n_similar_gages = st.sidebar.slider('Number of Similar Gages:', min_value=1, max_value=20, value=5)

window_start_str = window_start.strftime('%Y-%m-%d')
window_end_str = window_end.strftime('%Y-%m-%d')

if st.sidebar.button("Run Dynamic Window"):
    dynamic_output_df = dynamic_window(geometries_for_gages_df, flow_df, features, siteid, window_start_str, window_end_str, n_similar_gages)
    st.session_state['dynamic_output_df'] = dynamic_output_df 
    st.write("Dynamic Window Output:", dynamic_output_df)


def get_top_n_gages(gage_target, top_n=2):

    closest_gages = similarity_sort(ids, 11316605)
    closest_gages = closest_gages[closest_gages['prop_recorded'] >= 0.50]
    closest_gages = closest_gages.drop(columns=['index'])
    closest_gages = pd.merge(closest_gages, merged_gdf, on='siteid', how='inner')
    closest_gages[['gage_id', 'siteid','similarity_score','latitude','longitude']]
    closest_gages_n = closest_gages.iloc[0:top_n]


    return closest_gages_n


def create_map(gage_target, top_n=2):
    if dynamic_output_df.empty:
        st.warning("No data available. Please run the dynamic window analysis.")
        return None

    start_date = pd.to_datetime(window_start)
    end_date = pd.to_datetime(window_end)

    date_range = (start_date, end_date)

    top_gages = get_top_n_gages(gage_target, top_n)

    our_target = merged_gdf[merged_gdf['siteid'] == gage_target]


    fig = px.scatter_mapbox(top_gages, lat='latitude', lon='longitude', 
                            hover_name='siteid', 
                            color_discrete_sequence=['darkblue'],  # Setting circles to dark blue
                            zoom=3)
    
    fig.add_scattermapbox(lat=our_target['latitude'], lon=our_target['longitude'],
                          hoverinfo='text',
                          text=our_target['siteid'],
                          marker=dict(size=12, color='red', opacity=0.8),  # You can adjust size and opacity as needed
                          name='Target Gage')
    
    fig.update_layout(mapbox_style='open-street-map',
                      mapbox=dict(center=dict(lat=our_target['latitude'].iloc[0], 
                                              lon=our_target['longitude'].iloc[0]),  # Center map on target gage
                                  zoom=10),  
                      showlegend=True)
    
    return fig

if not dynamic_output_df.empty:
    map = create_map(siteid, n_similar_gages)
    st.plotly_chart(map)
else:
    st.write("Please generate data using the dynamic window settings.")
