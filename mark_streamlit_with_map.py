import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

def load_data(csv_name):
    data = pd.read_csv(csv_name)
    return data

# Load dynamic output data and geometries
dynamic_output_df = load_data('nathan_dynamic_window_output.csv')
geometries_for_gages_df = load_data('features_df.csv')

# Sidebar: Date Range Selection
st.sidebar.header('Select Date Range:')
date_range = st.sidebar.date_input('Date range', [pd.to_datetime(dynamic_output_df['date']).min(), pd.to_datetime(dynamic_output_df['date']).max()])

# Function to create map
@st.cache_data
def create_map(date_range):
    # Convert date_range to datetime objects
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    map = folium.Map(location=[geometries_for_gages_df['latitude'].mean(), geometries_for_gages_df['longitude'].mean()], zoom_start=8)

    # Filter dynamic output data for selected date range
    filtered_df = dynamic_output_df[(pd.to_datetime(dynamic_output_df['date']) >= start_date) & 
                                    (pd.to_datetime(dynamic_output_df['date']) <= end_date)]

    # Get unique gage IDs within the date range
    unique_gages = set(sum(filtered_df['ref_gages'].apply(eval).tolist(), []))

    # Add circle for each gage
    for idx, gage_id in enumerate(unique_gages):
        gage_data = geometries_for_gages_df[geometries_for_gages_df['siteid'] == gage_id]
        if not gage_data.empty:
            folium.Circle(location=[gage_data.iloc[0]['latitude'], gage_data.iloc[0]['longitude']], 
                          radius=1000,  # Adjust radius as needed
                          popup=str(gage_id),
                          fill=True,
                          color='blue',  # Change circle color if needed
                          fill_color='blue').add_to(map)

    return map

# Main content area
map = create_map(date_range)
st_folium(map, width=700, height=500)
