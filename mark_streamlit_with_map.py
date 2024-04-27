import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
from copy_of_sooyeon_manual_similarity_measure_expert_rescale_rfe import *

def load_data(csv_name):
    data = pd.read_csv(csv_name)
    return data

# Load data
dynamic_output_df = load_data('nathan_dynamic_window_output.csv')
geometries_for_gages_df = load_data('features_df.csv')

# Dummy data for testing which has different n gages recommended for each date
data = """
date,ref_gages
2010-12-20,"[11414100, 11408000, 11407815, 11416000, 103366097, 11414410, 9429490]"
2010-12-21,"[11414700, 11414100, 11408000, 11407815, 11416000, 11414410]"
2010-12-22,"[11414700, 11414100]"
2010-12-23,"[11414100, 11408000, 11407815, 11416000, 11414410, 9429490]"
2010-12-24,"[11414700, 11414100, 11408000, 11407815, 11416000, 103366097]"
2010-12-25,"[11414700, 11414100, 11407815, 11416000, 11414410]"
2010-12-26,"[11414700, 11414100, 11408000, 11407815, 11416000]"
2010-12-27,"[11414700, 11408000, 11407815, 11416000, 11414410]"
2010-12-28,"[11414700, 11414100, 11408000, 11407815, 11416000]"
2010-12-29,"[11414700, 11408000, 11407815, 11416000, 9429490]"
2010-12-30,"[11414700, 11414100, 11408000, 11407815, 11416000]"
2010-12-31,"[11414700, 11414100, 11251000, 11408000, 11407815, 103366097]"
"""

# Read the text data into a DataFrame
df = pd.read_csv(StringIO(data))

# Sidebar: Date Range Selection
st.sidebar.header('Select Date Range:')
date_range = st.sidebar.date_input('Date range', [pd.to_datetime(dynamic_output_df['date']).min(), pd.to_datetime(dynamic_output_df['date']).max()])

# Get unique gages from data_df
unique_gages = set(sum(df['ref_gages'].apply(eval).tolist(), []))
num_unique_gages = len(unique_gages)
top_n = st.sidebar.slider('Select number of gages:', min_value=1, max_value=num_unique_gages, value=2)

# Select target gage
target_gage = st.sidebar.selectbox('Select target gage:', unique_gages)

def get_top_n_gages(data, date_range, top_n=2):
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    filtered_df = data[(pd.to_datetime(data['date']) >= start_date) & 
                       (pd.to_datetime(data['date']) <= end_date)]

    unique_gages = pd.Series(sum(filtered_df['ref_gages'].apply(eval).tolist(), [])).value_counts().head(top_n)

    

    print(unique_gages)
    return unique_gages

# Function to create Plotly map
def create_map(date_range, top_n=2, target_gage=None):
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1] if len(date_range) > 1 else date_range[0])

    filtered_df = df[(pd.to_datetime(df['date']) >= start_date) & 
                     (pd.to_datetime(df['date']) <= end_date)]

    top_gages = get_top_n_gages(filtered_df, date_range, top_n)

    fig = go.Figure()

    for gage_id, count in top_gages.items():
        gage_data = geometries_for_gages_df[geometries_for_gages_df['siteid'] == gage_id]
        if not gage_data.empty:
            if gage_id == target_gage:
                color = 'blue'  # TARGET GAGE
            else:
                color = 'red'   # REF GAGES
            fig.add_trace(go.Scattermapbox(
                lat=gage_data['latitude'],
                lon=gage_data['longitude'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=9,
                    color=color,
                    opacity=0.7
                ),
                hoverinfo='text',
                text=gage_data['siteid'],
                name=f"Site ID: {gage_id}, Frequency: {count}"
            ))

    fig.update_layout(
        hovermode='closest',
        mapbox=dict(
            accesstoken='pk.eyJ1IjoibXJraWViYXJyIiwiYSI6ImNsb2Rxbm1tZzA0aHUyeHBmMGRhY2NtZ3UifQ.sfAJYCoOIo6s4Zm6-PrmJg',
            style='outdoors',  # Choose the mapbox style here
            center=dict(
                lat=geometries_for_gages_df['latitude'].mean(),
                lon=geometries_for_gages_df['longitude'].mean()
            ),
            zoom=8
        )
    )

    if target_gage:
        target_data = geometries_for_gages_df[geometries_for_gages_df['siteid'] == target_gage]
        if not target_data.empty:
            fig.update_layout(
                mapbox=dict(
                    center=dict(
                        lat=target_data['latitude'].iloc[0],
                        lon=target_data['longitude'].iloc[0]
                    ),
                    zoom=10
                )
            )

    return fig

st.title("Stream GageIDs Map")
map = create_map(date_range, top_n, target_gage)
st.plotly_chart(map)
