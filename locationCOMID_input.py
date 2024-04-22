import streamlit as st
import pandas as pd
# from geopy.geocoders import Nominatim

# def get_gages_by_(df, user_location):
#     def geocode_location(location):
#     geolocator = Nominatim(user_agent="streamlit_app")
#     location = geolocator.geocode(location)
#     if location:
#         return (location.latitude, location.longitude)
#     else:
#         return (None, None)
#     # Geocode the user location
#     lat, lon = geocode_location(user_location)
#     if lat is None or lon is None:
#         return None, "Unable to geocode the location."

#     # Calculate distances from user location to all sites in the DataFrame
#     df['distance'] = ((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)**0.5

#     # Get the row with the smallest distance
#     nearest_site = df.loc[df['distance'].idxmin()]
#     return nearest_site['site_id'], nearest_site['sitename']

def get_gages_by_comid(df, comid):

    if comid in df['comid_medr'].values:
        # Group by 'comid_medr' and get the list of 'gage_id'
        gage_ids = df[df['comid_medr'] == comid]['gage_id'].unique().tolist()
        return gage_ids
    else:
        st.error('The entered COMID is not found.', icon="🚨")
        return []