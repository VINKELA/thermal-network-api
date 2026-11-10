import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point

def extract_nearby_buildings(input_csv, output_csv, target_lat, target_lon, radius_m=1500):
    print(f"Reading data from {input_csv}...")
    
    # 1. Load data and sanitize columns to lowercase
    # This turns 'BUILDINGID' into 'buildingid', 'Heat_KW' into 'heat_kw', etc.
    df = pd.read_csv(input_csv, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. Convert WKT strings to Shapely geometry objects
    df['geometry'] = df['geometry'].apply(wkt.loads)
    
    # 3. Create GeoDataFrame and set CRS (UTM Zone 17N)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=32617, inplace=True, allow_override=True)
    
    # 4. Filter for RESIDENTIAL buildings
    if 'buildingty' in gdf.columns:
        residential_gdf = gdf[gdf['buildingty'].str.upper() == 'RESIDENTIAL'].copy()
    else:
        residential_gdf = gdf.copy()
        print("Warning: 'buildingty' column not found. Processing all buildings.")

    # 5. Define target location and reproject to UTM (Meters)
    target_point_wgs84 = gpd.GeoSeries([Point(target_lon, target_lat)], crs="EPSG:4326")
    target_point_utm = target_point_wgs84.to_crs(epsg=32617).iloc[0]
    
    # 6. Calculate exact distances, filter to radius, and sort to get Top 100
    residential_gdf['distance_to_poi'] = residential_gdf.geometry.distance(target_point_utm)
    
    nearby_buildings = residential_gdf[residential_gdf['distance_to_poi'] <= radius_m].copy()
    
    # Sort by distance (closest first) and take the top 100
    top_100_buildings = nearby_buildings.sort_values('distance_to_poi').head(10000)
    
    print(f"Found {len(nearby_buildings)} residential buildings within {radius_m}m. Keeping the closest {len(top_100_buildings)}.")
    
    # 7. Convert filtered buildings back to Lat/Lon for the output
    nearby_buildings_wgs84 = top_100_buildings.to_crs(epsg=4326)
    centroids = nearby_buildings_wgs84.geometry.centroid
    
    # 8. Map to the EXACT requested output format using your specific dataset headers
    output_df = pd.DataFrame()
    output_df['latitude'] = centroids.y
    output_df['longitude'] = centroids.x
    
    output_df['name'] = nearby_buildings_wgs84.get('buildingid', '')
    
    # UPDATED: Now pulling from your newly created Heat_KW column
    output_df['heating_demand'] = nearby_buildings_wgs84.get('heat_kw', '')
    
    output_df['cooling_demand'] = ''
    output_df['waste_heat_source_type'] = ''
    output_df['floor_area'] = nearby_buildings_wgs84.get('effarea', '') 
    output_df['height'] = nearby_buildings_wgs84.get('storys', '') 
    output_df['fsa'] = ''
    output_df['area'] = nearby_buildings_wgs84.get('shapearea', '')
    output_df['type'] = 'residential'
    output_df['load_type'] = ''
    
    # 9. Create a row for the Point of Interest itself
    poi_data = {
        'latitude': [target_lat],
        'longitude': [target_lon],
        'name': ['Point of Interest'],
        'heating_demand': [''],
        'cooling_demand': [''],
        'waste_heat_source_type': [''],
        'floor_area': [''],
        'height': [''],
        'fsa': [''],
        'area': [''],
        'type': ['poi'],  # Tagging it as the POI for easy identification
        'load_type': ['']
    }
    poi_df = pd.DataFrame(poi_data)
    
    # 10. Combine the POI row with the top 100 buildings (POI goes at the very top)
    final_output_df = pd.concat([poi_df, output_df], ignore_index=True)
    
    # 11. Save the final data to a new CSV file
    final_output_df.to_csv(output_csv, index=False)
    print(f"Successfully saved formatted data to {output_csv}")

# ==========================================
# Run the Function
# ==========================================
if __name__ == "__main__":
    # Target coordinates
    LATITUDE = 43.3540728
    LONGITUDE = -79.8023175
    
    # File paths
    INPUT_FILE = r"C:\Users\orjik\source\research\thermal-network-api\data\Husni\Burlington_NewModel_GoWest_MattModel.csv"
    OUTPUT_FILE = r"C:\Users\orjik\source\research\thermal-network-api\data\Buildinguz\filtered_residential_top1004.csv"
    
    # Execute
    extract_nearby_buildings(
        input_csv=INPUT_FILE, 
        output_csv=OUTPUT_FILE, 
        target_lat=LATITUDE, 
        target_lon=LONGITUDE
    )