API_KEY = "-"
import pandas as pd
import requests
import time
import pickle


df = pd.read_csv('housing.csv')

cache_file = "geocoding_cache.pkl"
try:
    with open(cache_file, "rb") as f:
        geocode_cache = pickle.load(f)
except FileNotFoundError:
    geocode_cache = {}

geo_info_dict = {}
def get_geo_info(lat, long):
    key = (lat, long)
    if key in geocode_cache:
        return geocode_cache[key]

    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{long}&key={API_KEY}"
        response = requests.get(url)
        result = response.json()

        if response.status_code == 200 and 'results' in result and len(result['results']) > 0:
            return result
        geocode_cache[key] = None
        return None
    except Exception as e:
        print(f"Error for ({lat}, {long}): {e}")
        return None


batch_size = 100
start_index = 0

for i in range(start_index, len(df), batch_size):
    batch = df.iloc[i:i + batch_size]
    batch_results = []

    for idx, row in batch.iterrows():
        geo_info = get_geo_info(row['latitude'], row['longitude'])
        geo_info_dict[idx] = geo_info
        batch_results.append(geo_info)
        time.sleep(0.1)

    with open(cache_file, "wb") as f:
        pickle.dump(geocode_cache, f)

    print(f"Processed {i + len(batch)} of {len(df)} rows")