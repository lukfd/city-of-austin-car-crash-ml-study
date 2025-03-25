import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np

def create_folium_density_heatmap(lats, longs, zoom_start=12):
    """
    Creates a Folium heatmap where the intensity of red represents the density of points.

    Args:
        lats: latitudes.
        longs: longitudes.
        zoom_start: Initial zoom level of the map.
    """

    if not lats or not longs or len(lats) != len(longs):
        print("Error: Invalid latitude or longitude lists.")
        return None

    try:
        # Calculate the map center
        map_center = [np.mean(lats), np.mean(longs)]

        # Create a DataFrame from the lat-long data
        data = pd.DataFrame({'lat': lats, 'lon': longs})

        # Create the Folium map
        m = folium.Map(location=map_center, zoom_start=zoom_start)

        # Prepare data for the HeatMap plugin
        heatmap_data = [[row['lat'], row['lon']] for index, row in data.iterrows() if pd.notna(row['lat']) and pd.notna(row['lon'])]

        # Create and add the HeatMap layer
        HeatMap(heatmap_data).add_to(m)

        return m

    except TypeError:
        print("Error: lats and longs must be numerical.")
        return None