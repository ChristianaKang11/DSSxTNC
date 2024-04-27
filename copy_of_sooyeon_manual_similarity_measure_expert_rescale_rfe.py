import numpy as np
import pandas as pd
import geopandas as gpd
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import numpy.linalg as nla
import sys
import math

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import geopandas as gpd

from datetime import datetime, timedelta
import random

# Contains stream metadata linked to site IDs (from StreamStats)
features = pd.read_csv('features_df.csv')

# Contains flow data linked to site IDs from January 2010 to December 2022 (Kirk's script, pulls from USGS with ulmo)
flow_df = pd.read_csv('flow_df.csv')

# Contains geopandas geometry (location) data
geodf = pd.read_csv('geodf.csv')

features['siteid_str'] = features['siteid'].astype(str)
geodf['siteid_str'] = geodf['siteid'].astype(str)

# Merge the metadata and geometry tables along site ID
merged = features.merge(geodf[['siteid_str', 'geometry']], left_on='siteid_str', right_on='siteid_str')
merged_gdf = gpd.GeoDataFrame(merged.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0_x']))

"""# EDA

flow_df: Represents a DataFrame containing flow data, which could include information such as flow values, timestamps (datetime), and gage IDs. Each row in this DataFrame  corresponds to a specific measurement of flow at a certain time and from a certain gage.

features_df: Represents a DataFrame containing additional features or variables that are relevant for expert rescaling. These features could include various characteristics of the gages or environmental factors that might influence flow measurements. Each row in this DataFrame  corresponds to a specific observation of these features, potentially with columns representing different attributes or measurements.
"""

merged_gdf.head()

# Drop more unnecessary columns
merged_gdf.drop(["0", "index_x", "index_y","Unnamed: 0_y",], axis=1)
merged_gdf.head()

def id_overlap(flow_df, features_df):
  flow_ids = pd.Series(flow_df['gage_id'].value_counts().index)
  features_ids = pd.Series(features_df['siteid'].value_counts().index)
  return flow_ids[flow_ids.isin(features_ids)]

  # TEST: MATCHES EXPECTATIONS
  # overlap_ids = []
  # for i in flow_ids:
  #   if i in features_ids:
  #     overlap_ids.append(i)
  # return overlap_ids

# Generate list of gage IDs that appear in both flow_df and features
id_overlap_list = id_overlap(flow_df, features).tolist()

def get_overlap_dfs(flow_df, features_df):
  overlap_id = id_overlap(flow_df, features_df)
  cleaned_flow = flow_df[flow_df['gage_id'].isin(overlap_id)]
  cleaned_features = features_df[features_df['siteid'].isin(overlap_id)]
  return cleaned_flow, cleaned_features

yip, yay = get_overlap_dfs(flow_df, features)
yipseries=yip["datetime"]

def date_overlap(flow_df, ref_id, target_id):
  """
  Returns a Series with the list of dates on which ref_id and target_id were both collecting data
  """

  ref_rows = flow_df[flow_df['gage_id'] == int(ref_id)]  # Compare 'gage_id' directly with ref_id as integer
  target_rows = flow_df[flow_df['gage_id'] == int(target_id)]  # Compare 'gage_id' directly with target_id as integer

  ref_date_overlap = ref_rows[ref_rows['datetime'].isin(target_rows['datetime'])]
  target_date_overlap = target_rows[target_rows['datetime'].isin(ref_rows['datetime'])]

  date_series = ref_date_overlap['datetime']

  return date_series

"""## Weight *Assignment*"""

id_overlap_list = id_overlap(flow_df, features).tolist()

flow_df_filtered = flow_df.rename(columns={'gage_id': 'siteid'})

# Filter the dataframes to only include rows with IDs in id_overlap_list
flow_df_filtered = flow_df_filtered[flow_df_filtered['siteid'].isin(id_overlap_list)]
features_filtered = features[features['siteid'].isin(id_overlap_list)]

feature_selection = features_filtered[['siteid', 'DRNAREA', 'LAKEAREA', 'ELEV', 'RELIEF', 'FOREST', 'PRECIP']]

merged_df = pd.merge(flow_df_filtered, feature_selection, on='siteid', how='left').dropna()
merged_df = merged_df[['value', 'siteid', 'DRNAREA', 'LAKEAREA', 'ELEV', 'RELIEF', 'FOREST', 'PRECIP']]
merged_df = merged_df.groupby('siteid').mean()

X_train = merged_df[['DRNAREA', 'LAKEAREA', 'ELEV', 'RELIEF', 'FOREST', 'PRECIP']]
y_train = merged_df['value']


model = LinearRegression()
model.fit(X_train, y_train)

coefficients = model.coef_

normalized_coefficients = coefficients / coefficients.sum()

weighted_features = X_train * normalized_coefficients.reshape(1, -1)
weighted_features

"""# Manual Similarity Measure"""

# We treat features of gages as vector and calculate their distance in vector space. The closer the distance, the similar the gages are.
# dropping nan values in selected features
def drop_nan(df, features):
  return df.dropna(subset=features)

def haversine(lat1, lon1, lat2, lon2):
    # convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # radius of Earth in miles. need to use use 6371 for km
    r = 3956

    return c * r

# normalizing features because using different scales
def normalize_features(df, features):
    for feature in features:
       df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()

# take in feature dictionary
# normalize the features
# new columns are original column name + _norm
# new dataframe with just the normalized features and ids
# return similarity score

def get_similarity_measure(df, feature_dictionary, gage_id):
    """
    calculating the weighted euclidean similarity

    parameters:
    - df: dataFrame containing the gauge data with normalized features
    - feature_names: array of feature names (columns in df) to be used for similarity
    - feature_weights: Dictionary with feature names as keys and their weights as values
    - random_gauge_features: Series or row from df for the randomly selected gauge

    returns:
    EDIT: should return a data frame with
    - series containing the weighted similarity scores for all gauges
    """

    # backup of the original data
    feature_names = list(feature_dictionary.keys())

    df = df.copy()
    df = drop_nan(df, feature_names)

    ref_lat = df.loc[df['siteid'] == gage_id, 'latitude'].values[0]
    ref_lon = df.loc[df['siteid'] == gage_id, 'longitude'].values[0]

    df['distance_to_ref'] = df.apply(lambda row: haversine(ref_lat, ref_lon, row['latitude'], row['longitude']), axis=1)

    feature_dictionary['distance_to_ref'] = 0.65

    feature_names += ['distance_to_ref']
    normalize_features(df, feature_names)

    # apply weights to the features
    df[feature_names] = df[feature_names].multiply([feature_dictionary[feature] for feature in feature_names], axis=1)

    # Extract the features for the reference gauge, ensuring it's a Series for easy subtraction
    reference_features = df.loc[df['siteid'] == gage_id, feature_names].iloc[0]

    # Calculate the Euclidean distance (similarity score) between the reference gauge and all others
    # Subtract the reference gauge features from each row, square, sum across columns, and square root
    df['similarity_score'] = df[feature_names].apply(lambda x: np.sqrt(((x - reference_features) ** 2).sum()), axis=1)

    # changed this to table of siteids and similarity score so we can have match row lengths when we filter based on only needed ids
    return df[['siteid', 'similarity_score']]

features_weighted_dict = {'DRNAREA': 0.005, 'LAKEAREA': 0.64,	'PRECIP': 0.37}
similarity_df = get_similarity_measure(merged_gdf, {'DRNAREA': 0.005, 'LAKEAREA': 0.64,	'PRECIP': 0.37}, 103087853)

# now we merge this to nathan's dynamic window
similarity_df.sort_values('similarity_score', ascending=False)

"""# Dynamic Windowing"""

# sizes = flow_df.groupby('gage_id').size().value_counts()
counts = flow_df.groupby('gage_id').count()[['value']]

counts['prop_recorded'] = counts['value'] / 4748 # 4748 days between January 1, 2010 and December 31, 2022
zero_flows = flow_df[flow_df['value'] == 0].groupby('gage_id').count()[['value']]
zero_flows['prop_zero'] = zero_flows['value'] / 4748

def id_overlap(flow_df, features_df):
  """
  Generates a series of all IDs that appear in both flow_df and features_df
  """
  # counts = flow_df.groupby('gage_id').count()[['value']]
  flow_ids = pd.Series(flow_df['gage_id'].value_counts().index)
  features_ids = pd.Series(features_df['siteid'].value_counts().index)
  return flow_ids[flow_ids.isin(features_ids)]

  # TEST: MATCHES EXPECTATIONS
  # overlap_ids = []
  # for i in flow_ids:
  #   if i in features_ids:
  #     overlap_ids.append(i)
  # return overlap_ids

id_overlap(flow_df, features)

# Generate list of gage IDs that appear in both flow_df and features
id_overlap_series = id_overlap(flow_df, features).tolist()

def date_overlap(flow_df, ref_id, target_id):
  """
  Returns a Series with the list of dates on which ref_id and target_id were both collecting data
  """

  ref_rows = flow_df[flow_df['gage_id'] == ref_id] # all records with gage_id corresponding to ref_id
  target_rows = flow_df[flow_df['gage_id'] == target_id] # all records with gage_id corresponding to target_id

  ref_date_overlap = ref_rows[ref_rows['datetime'].isin(target_rows['datetime'])]
  target_date_overlap = target_rows[target_rows['datetime'].isin(ref_rows['datetime'])]

  date_series = ref_date_overlap['datetime']
  return date_series

date_overlap(flow_df, random.choice(id_overlap_series), random.choice(id_overlap_series))

def threshold(flow_df, target_id):
  # copy = flow_df.copy()
  copy = pd.DataFrame(id_overlap_series).rename({0: 'gage_id'}, axis=1)
  # copy['overlap_prop'] = copy.apply(lambda row: len(date_overlap(flow_df, target_id, row['gage_id'])) / 4748)
  copy['overlap_prop'] = copy.apply(lambda row: len(date_overlap(flow_df, target_id, row['gage_id'])) / 4748, axis=1)
  return copy

overlap_prop_df = threshold(flow_df, 9429490)

new_counts = counts.reset_index()

ids = pd.DataFrame(id_overlap_series).rename({0: 'gage_id'}, axis=1)
mod_geodf = geodf.copy()
mod_geodf['siteid_str'] = mod_geodf['siteid'].astype(str)
mod_geodf = mod_geodf[mod_geodf['siteid_str'].str.contains(r'^\d+$')].drop(columns='siteid_str')
mod_geodf['siteid'] = mod_geodf['siteid'].astype(int)
ids = gpd.GeoDataFrame(ids.merge(mod_geodf[['siteid', 'geometry']], left_on='gage_id', right_on='siteid')
                          .merge(new_counts[['gage_id', 'prop_recorded']], on='gage_id'))
ids

def nearest_points(geodf, siteid, n=1):
  """
  Find the n nearest points in a GeoDataFrame to the point specified by target_index.

  Parameters:
  - geodf: A GeoDataFrame containing points.
  - target_index: The index of the target point in geodf.
  - n: The number of nearest points to return, including the target point.

  Returns:
  - A GeoDataFrame with an added 'color' column.
  """

  # Extract the target point using the provided index
  target_point = geodf.loc[geodf['siteid_str'] == siteid].iloc[0]['geometry']

  print(target_point)

  # Calculate the distances from target_point to all points in geodf
  geodf['distances'] = geodf.geometry.distance(target_point)

  # Sort the GeoDataFrame by distance
  closest = geodf.nsmallest(n + 1, 'distances')

  # Add a color column
  # geodf['color'] = 0  # default color
  # geodf.loc[closest.index, 'color'] = 1 # color for the nearest points
  # geodf.loc[target_index,'color'] = 2  # default color

  return closest

def nearest_algorithm(geodf, siteid, threshold=0.0, n=1):
  """
  Find the n nearest reference gages in the GeoDataFrame to the target point specified by siteid that meet the threshold of recording.

  Parameters:

  Returns:
  - DataFrame with the n nearest reference gages to the target location that meet the threshold proportion of recorded dates.
  """
  # Extract target point location from given siteid
  # FOR FUTURE USE: target_point can be set to any latitude/longitude location
  # Instead of using a specific siteid, use shapely.point to define Point([x, y])
  # Calculate the distances from target_point to all points in geodf
  # Calculate the distances from target_point to all points in geodf
  copydf = geodf.copy()
  copydf = copydf.set_geometry('geometry')

  target_point = copydf[copydf['siteid'] == siteid]['geometry'].iloc[0]

  copydf['distance'] = copydf.geometry.distance(target_point)
  
  # Sort the DataFrame by distance in descending order
  copydf = copydf.sort_values('distance',ascending=True)

  closest_ids = []
  closest_distances = []
  closest_props = []

  
  for gage in copydf['siteid']:
    gage_row = copydf[copydf['siteid'] == gage]

    if gage_row['distance'].iloc[0] == 0: # distance == 0 or less than some amount
      # skip this one if the two siteids refer to the same gage
      continue
    elif len(closest_ids) < n:
      if gage_row['prop_recorded'].iloc[0] >= threshold:
        closest_ids.append(gage)
        closest_distances.append(gage_row['distance'].iloc[0])
        closest_props.append(gage_row['prop_recorded'].iloc[0])
      else:
        # if threshold prop_recorded is not met, skip this one
        continue
    else:
      return pd.DataFrame({'siteid': closest_ids, 'distance': closest_distances, 'prop_recorded': closest_props})

id_overlap_series = id_overlap(flow_df, features).tolist()

date_list = pd.DataFrame(flow_df.groupby('gage_id').agg({'datetime': lambda x: list(x)}))
overlap_df = pd.DataFrame(id_overlap_series).rename({0: 'gage_id'}, axis=1)
compiled_dates = overlap_df.merge(date_list, on='gage_id').rename({'datetime': 'date_list'}, axis=1)
# compiled_dates['list_prop'] = compiled_dates.apply(lambda row: len(row['date_list']) / 4748, axis=1)

ids_copy = ids.copy()
new_df = ids_copy.merge(compiled_dates, on='gage_id')

def ids_table(geodf, flowdf, featuresdf):

  # find the gage ids that exist in both flowdf and featuresdf
  id_overlap_series = id_overlap(flowdf, featuresdf).tolist()

  # create the ids table: gage_id/siteid, geometry
  ids = pd.DataFrame(id_overlap_series).rename({0: 'gage_id'}, axis=1)
  mod_geodf = geodf.copy()
  mod_geodf['siteid_str'] = mod_geodf['siteid'].astype(str)
  mod_geodf = mod_geodf[mod_geodf['siteid_str'].str.contains(r'^\d+$')].drop(columns='siteid_str')
  mod_geodf['siteid'] = mod_geodf['siteid'].astype(int)
  ids = gpd.GeoDataFrame(ids.merge(mod_geodf[['siteid', 'geometry']], left_on='gage_id', right_on='siteid'))
  return ids

def window_range(window_start, window_end):
    """
    Generate a list of dates between window_start and window_end.
    
    Parameters:
    - window_start (datetime or str): Start date of the window in the format 'YYYY-MM-DD' or datetime object.
    - window_end (datetime or str): End date of the window in the format 'YYYY-MM-DD' or datetime object.
    
    Returns:
    - List of dates between window_start and window_end.
    """
    # If window_start or window_end are strings, convert them to datetime objects
    if isinstance(window_start, str):
        window_start = datetime.strptime(window_start, '%Y-%m-%d').date()
    else:
        window_start = window_start.date()  # Assuming window_start is a datetime object

    if isinstance(window_end, str):
        window_end = datetime.strptime(window_end, '%Y-%m-%d').date()
    else:
        window_end = window_end.date()  # Assuming window_end is a datetime object
    
    return [window_start + timedelta(days=i) for i in range((window_end - window_start).days + 1)]

def similarity_sort(ids_table, siteid):
    """
    Find the n nearest reference gages in the GeoDataFrame to the target point specified by siteid that meet the threshold of recording.

    Takes input_df = ids_table(geodf, flowdf, featuresdf) or similar

    Returns:
    - DataFrame with the n nearest reference gages to the target location that meet the threshold proportion of recorded dates.
    """

    # generate ids table: gage id, geometry

    # Extract target point location from given siteid
    # FOR FUTURE USE: target_point can be set to any latitude/longitude location
    # Instead of using a specific siteid, use shapely.point to define Point([x, y])


    target_point = ids_table.loc[ids_table['siteid'] == siteid].iloc[0]['geometry']
 

    # Calculate the distances from target_point to all points in geodf
    copydf = ids_table.copy()

    similarity_df = get_similarity_measure(merged_gdf, {'DRNAREA': 0.005, 'LAKEAREA': 0.64, 'PRECIP': 0.37}, siteid)

    merged_similarity_ids_df = pd.merge(copydf, similarity_df, on='siteid', how='inner').dropna(subset=['siteid'])
    merged_similarity_ids_df = merged_similarity_ids_df.sort_values('similarity_score', ascending=False)
    merged_similarity_ids_df = merged_similarity_ids_df.reset_index()
    merged_similarity_ids_df = merged_similarity_ids_df[merged_similarity_ids_df['siteid'] != siteid]

    return merged_similarity_ids_df

def dynamic_window(geodf, flowdf, featuresdf, siteid, window_start='', window_end='', n=1):

  """
  For each day between window_start and window_end, returns a DataFrame containing the n most similar reference gages to a given target.

  Parameters:
  - geodf (DataFrame): contains shape data for gages
  - flowdf (DataFrame): contains flow data for gages
  - featuresdf (DataFrame): contains data about characteristics of the gage and its location
  - siteid (int): siteid/gage_id of the target gage
  - window_start (str): start date of window (YYYY-MM-DD)
  - window_end (str): end date of window (YYYY-MM-DD)
  - n (int): number of similar gages desired for each date

  Returns:
  - DataFrame containing `date` (all dates in window range) and `ref_gages` (n similar gages for each date)
  """

  # find the gage ids that exist in both flowdf and featuresdf
  id_overlap_series = id_overlap(flowdf, featuresdf).tolist()

  # generate ids table: gage id, geometry
  ids = ids_table(geodf, flowdf, featuresdf)
  
    # create compiled_dates: gage_id, date_list (which dates each gage was recording on)
  date_list = pd.DataFrame(flowdf.groupby('gage_id').agg({'datetime': lambda x: list(x)}))
  overlap_df = pd.DataFrame(id_overlap_series).rename({0: 'gage_id'}, axis=1)
  compiled_dates = overlap_df.merge(date_list, on='gage_id').rename({'datetime': 'date_list'}, axis=1)

  # merge ids and compiled_dates
  new_df = ids.merge(compiled_dates, on='gage_id')

  # sort by similarity score
  closest_gages = similarity_sort(new_df, siteid)
  closest_gages = closest_gages.drop(columns=['index'])

  # determine all dates between window_start and window_end
  window_dates = window_range(window_start, window_end)
  window_dates = [date.strftime('%Y-%m-%d') for date in window_dates]

  # for each date in window_dates, find the n most similar gages that were recording
  similar_gages_all = []
  for date in window_dates:
    similar_gages = []
    i = 0
    while len(similar_gages) < n:
      ith_closest_gage = closest_gages.iloc[i]

      if date in ith_closest_gage['date_list']:
        similar_gages.append(ith_closest_gage['gage_id'])
      i += 1

    similar_gages_all.append(similar_gages)

  # construct DataFrame linking window_dates to similar_gages_all for convenience
  window_gages = pd.DataFrame({'date': window_dates, 'ref_gages': similar_gages_all})

  return window_gages

meowy = dynamic_window(merged_gdf, flow_df, features, 11520500, window_start='2020-12-10', window_end='2020-12-31', n=5)