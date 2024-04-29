
import numpy as np
import pandas as pd
import geopandas as gpd
import random
from datetime import datetime



# Contains stream metadata linked to site IDs (from StreamStats)
features = pd.read_csv('/content/drive/MyDrive/StreamSage_2024_DSS_TNC/Data_2024/stream_stats_wtsh.csv')

# Contains flow data linked to site IDs from January 2010 to December 2022 (Kirk's script, pulls from USGS with ulmo)
flow_df = pd.read_csv("/content/drive/MyDrive/StreamSage_2024_DSS_TNC/Data_2024/daily_flow_data_from_201001_202212.csv")

# Contains geopandas geometry (location) data
geodf = gpd.read_file('/content/drive/MyDrive/StreamSage_2024_DSS_TNC/Data_2024/Site locations/StreamGages_SB19_v3c.shp')

# Convert all site IDs to strings for matching
# In order to make all the datatype uniform
features['siteid_str'] = features['siteid'].astype(str)
geodf['siteid_str'] = geodf['siteid'].astype(str)

# Merge the metadata and geometry tables along site ID
merged = features.merge(geodf[['siteid_str', 'geometry']], left_on='siteid_str', right_on='siteid_str')
merged_gdf = gpd.GeoDataFrame(merged.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0_x']))

def id_overlap(flow_df, features_df):
  flow_ids = pd.Series(flow_df['gage_id'].value_counts().index)
  features_ids = pd.Series(features_df['siteid'].value_counts().index)
  return flow_ids[flow_ids.isin(features_ids)]

def get_overlap_dfs(flow_df, features_df):
  overlap_id = id_overlap(flow_df, features_df)
  cleaned_flow = flow_df[flow_df['gage_id'].isin(overlap_id)]
  cleaned_features = features_df[features_df['siteid'].isin(overlap_id)]
  return cleaned_flow, cleaned_features


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

    ref_lat = df.iloc[[0]]['latitude'].values[0]
    ref_lon = df.iloc[[0]]['longitude'].values[0]

    df['distance_to_ref'] = df.apply(lambda row: haversine(ref_lat, ref_lon, row['latitude'], row['longitude']), axis=1)

    feature_dictionary['distance_to_ref'] = -0.74 #Assign negative weight to distance to favor closer reference gages

    feature_names += ['distance_to_ref']
    normalize_features(df, feature_names)

    # apply weights to the features
    df[feature_names] = df[feature_names].multiply([feature_dictionary[feature] for feature in feature_names], axis=1)

    # Extract the features for the reference gauge, ensuring it's a Series for easy subtraction
    reference_features = df.iloc[0]

    # Calculate the Euclidean distance (similarity score) between the reference gauge and all others
    # Subtract the reference gauge features from each row, square, sum across columns, and square root
    df['similarity_score'] = df[feature_names].apply(lambda x: np.sqrt(((x - reference_features) ** 2).sum()), axis=1)

    # changed this to table of siteids and similarity score so we can have match row lengths when we filter based on only needed ids
    return df[['siteid', 'similarity_score']]


"""# Dynamic Windowing"""
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


def threshold(flow_df, target_id):
  # copy = flow_df.copy()
  copy = pd.DataFrame(id_overlap_series).rename({0: 'gage_id'}, axis=1)
  # copy['overlap_prop'] = copy.apply(lambda row: len(date_overlap(flow_df, target_id, row['gage_id'])) / 4748)
  copy['overlap_prop'] = copy.apply(lambda row: len(date_overlap(flow_df, target_id, row['gage_id'])) / 4748, axis=1)
  return copy


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
  target_point = geodf.loc[geodf['siteid'] == siteid].iloc[0]['geometry']

  # Calculate the distances from target_point to all points in geodf
  copydf = geodf.copy()

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

def window_range(window_start='2010-01-01', window_end='2022-12-31'):
  date_start = datetime.strptime(window_start, '%Y-%m-%d').date()
  date_end = datetime.strptime(window_end, '%Y-%m-%d').date()

  date_list = []
  while date_start <= date_end:
    date_list.append(date_start.strftime('%Y-%m-%d'))
    date_start += timedelta(days=1)

  return date_list


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

  hi = nearest_algorithm(ids_table, siteid, threshold=0.8, n=50)
  hi = pd.merge(hi, merged_gdf, on='siteid', how='inner')

  similarity_df = get_similarity_measure(hi, {'LAKEAREA': 0.64,	'PRECIP': 0.80}, siteid) 

  merged_similarity_ids_df = pd.merge(copydf, similarity_df, on='siteid', how='inner').dropna(subset=['siteid'])

  merged_similarity_ids_df = merged_similarity_ids_df.sort_values('similarity_score', ascending=False)
  merged_similarity_ids_df = merged_similarity_ids_df.reset_index()

  # drop the row containing the target_point itself
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

  # determine all dates between window_start and window_end
  window_dates = window_range(window_start, window_end)

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



"""# Expert Rescale"""

def gage_stats(features_df, gage_id):
    gage_features = features_df.loc[features_df["siteid"] == gage_id]
    gage_map, gage_da = list(gage_features['PRECIP'])[0], list(gage_features["totdasqmi"])[0]
    return gage_map, gage_da

def expert_rescale(flow_df, features_df, gage_id, ref_id):
  # use this function for expert rescale
    flow, feat = get_overlap_dfs(flow_df, features_df)

    overlap_dates = date_overlap(flow, ref_id, gage_id)
    filteredflow = flow[flow["datetime"].isin(overlap_dates)]

    gage_flows = list(filteredflow[filteredflow["gage_id"].astype(int) == gage_id]["value"])
    ref_flows = list(filteredflow[filteredflow["gage_id"].astype(int) == ref_id]["value"])

    if(len(gage_flows) == 0 or len(ref_flows) == 0):
      print("No data for gages")
      return None, None, None

    ref_map, ref_da = gage_stats(feat, ref_id)
    gage_map, gage_da = gage_stats(feat, gage_id)

    if (ref_map == 0 or gage_map == 0) and not ref_da == 0:
      ret = np.asarray(gage_da/ref_da) * ref_flows
    elif (ref_da == 0 or gage_map == 0) and not ref_map == 0:
      ret = np.asarray(gage_map/ref_map) * ref_flows
    elif (ref_map == 0 and ref_da == 0) or (gage_map == 0 and gage_da == 0):
      ret = ref_flows
    else:
      ret = ref_flows * np.asarray(gage_da/ref_da) * np.asarray(gage_map/ref_map)

    # replacing NaN values in array with 0. In the future, could try imputation
    ret = np.where(np.isnan(ret), 0, ret)

    dates = list(filteredflow[filteredflow["gage_id"] == gage_id]["datetime"])

    result_df = pd.DataFrame({
        "predicted": ret,
        "observed": gage_flows,
        "date": dates,
        "ref_id": [ref_id] * len(dates)  # Repeat ref_id for the length of dates
    })

    return result_df

    #return ret, gage_flows, dates



"""# Dates and Predicted Flow Value

The final predictive function that calls dynamic windowing and expert rescale within it, and make target gage id, window_start, window_end as the parameters
"""

def predictedflow(target_id, window_start, window_end):
  target_gage_id = target_id
  target_map, target_da = gage_stats(features, target_gage_id)
  start_date = window_start
  end_date = window_end

  reference_gages = dynamic_window(geodf, flow_df, features, target_gage_id, window_start=start_date, window_end=end_date, n=3)

  flow_df_copy = flow_df.copy()
  flow_df_copy = flow_df_copy.loc[(flow_df_copy['datetime'] >= start_date) & (flow_df_copy['datetime'] <= end_date)]
  #Apply Expert Rescale 
  flow_series = []
  for date in reference_gages['date']:
    date_flows = []
    for ref in reference_gages[reference_gages['date'] == date].iloc[0, 1]:
      ref_map, ref_da = gage_stats(features, ref)
      ref_map = ref_map if ((ref_map != None) and (ref_map != np.nan)) else 0
      ref_da = ref_da if ((ref_da != None) and (ref_da != np.nan)) else 0
      map_scalar = (target_map / ref_map) if (ref_map != 0) else 1
      da_scalar = (target_da / ref_da) if (ref_da != 0) else 1
      ref_flow = flow_df_copy[(flow_df_copy['gage_id'] == ref) & (flow_df_copy['datetime'] == date)]['value']
      pred_flow = map_scalar * da_scalar * ref_flow
      date_flows.append(pred_flow)
    date_flows = np.array(date_flows)
    flow_series.append(date_flows)
  flow_series = np.array(flow_series)

  reference_gages['gage_1_flow'] = flow_series[:, 0]
  reference_gages['gage_2_flow'] = flow_series[:, 1]
  reference_gages['gage_3_flow'] = flow_series[:, 2]
  reference_gages['observed_flow'] = flow_df_copy[flow_df_copy['gage_id'] == target_gage_id]['value']
  reference_gages["predicted flow value"] = reference_gages[["gage_1_flow", "gage_2_flow", "gage_3_flow"]].mean(axis=1)
  output_df = reference_gages[["date", "predicted flow value"]]
  return output_df

