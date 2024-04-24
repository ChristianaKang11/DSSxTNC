import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from io import StringIO

def load_data(csv_name):
    data = pd.read_csv(csv_name)
    return data

# contains suggested gages from nathan's dynamic window
dynamic_output_df = load_data('nathan_dynamic_window_output.csv')

# contains latitude and longitude from features df
geometries_for_gages_df = load_data('features_df.csv')

# dummy data for testing which has diff n gages recommended for each date

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
target_gage = st.sidebar.selectbox('Select target gage:', unique_gages)

num_unique_gages = len(unique_gages)
top_n = st.sidebar.slider('Select number of gages:', min_value=1, max_value=num_unique_gages, value=2)


def get_top_n_gages(data, date_range, top_n=2):
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    filtered_df = data[(pd.to_datetime(data['date']) >= start_date) & 
                       (pd.to_datetime(data['date']) <= end_date)]

    unique_gages = pd.Series(sum(filtered_df['ref_gages'].apply(eval).tolist(), [])).value_counts().head(top_n)

    return unique_gages


# Function to create map
@st.cache_data
def create_map(date_range, top_n=2, target_gage=None):
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1] if len(date_range) > 1 else date_range[0])

    map = folium.Map(location=[geometries_for_gages_df['latitude'].mean(), geometries_for_gages_df['longitude'].mean()], zoom_start=8)

    if target_gage is not None and target_gage in geometries_for_gages_df['siteid'].tolist():
        target_data = geometries_for_gages_df[geometries_for_gages_df['siteid'] == target_gage]
        if not target_data.empty:
            map.location = [target_data.iloc[0]['latitude'], target_data.iloc[0]['longitude']]
            map.zoom_start = 10

    top_gages = get_top_n_gages(df, date_range, top_n)

    for gage_id, count in top_gages.items():
        gage_data = geometries_for_gages_df[geometries_for_gages_df['siteid'] == gage_id]
        if not gage_data.empty:
            color = 'red' if gage_id == target_gage else 'blue'
            folium.Circle(location=[gage_data.iloc[0]['latitude'], gage_data.iloc[0]['longitude']], 
                          radius=1000, 
                          popup=f"Site ID: {gage_id}, Frequency: {count}",
                          fill=True,
                          color=color,  
                          fill_color=color).add_to(map)

    return map



st.title("Stream GageIDs Map")
map = create_map(date_range, top_n, target_gage)
st_folium(map, width=700, height=500)