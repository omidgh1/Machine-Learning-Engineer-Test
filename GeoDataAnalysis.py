API_KEY = "-"
import pandas as pd
import requests
import time
import pickle


# Sample dataframe (replace with your actual data)
df = pd.read_csv('housing.csv')

# Cache to store already queried results
cache_file = "geocoding_cache.pkl"
try:
    with open(cache_file, "rb") as f:
        geocode_cache = pickle.load(f)
except FileNotFoundError:
    geocode_cache = {}

geo_info_dict = {}
# Function to get postal code using Google API
def get_geo_info(lat, long):
    key = (lat, long)
    if key in geocode_cache:
        return geocode_cache[key]

    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{long}&key={API_KEY}"
        response = requests.get(url)
        result = response.json()

        # Parse the postal code
        if response.status_code == 200 and 'results' in result and len(result['results']) > 0:
            return result
        geocode_cache[key] = None
        return None
    except Exception as e:
        print(f"Error for ({lat}, {long}): {e}")
        return None


# Batch processing
batch_size = 100  # Number of rows to process per batch
start_index = 0  # Resume from a specific index if needed

for i in range(start_index, len(df), batch_size):
    batch = df.iloc[i:i + batch_size]
    batch_results = []

    for idx, row in batch.iterrows():
        geo_info = get_geo_info(row['latitude'], row['longitude'])
        geo_info_dict[idx] = geo_info  # Store the result in dictionary using index as key
        batch_results.append(geo_info)
        time.sleep(0.1)  # Avoid hitting rate limits

    #df.loc[i:i + batch_size - 1, 'postal_code'] = batch_results

    # Save progress to cache file
    with open(cache_file, "wb") as f:
        pickle.dump(geocode_cache, f)

    print(f"Processed {i + len(batch)} of {len(df)} rows")

# Final