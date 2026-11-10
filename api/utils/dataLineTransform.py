import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point

def extract_top_5_closest_buildings(input_csv, output_csv, target_lat, target_lon):
    print(f"Reading data from {input_csv}...")
    
    # 1. Load data and sanitize columns to lowercase
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

    # 5. Define target point (little Montreal) and reproject to UTM (Meters)
    target_point_wgs84 = gpd.GeoSeries([Point(target_lon, target_lat)], crs="EPSG:4326")
    target_point_utm = target_point_wgs84.to_crs(epsg=32617).iloc[0]
    
    # ==========================================
    # CORE LOGIC: TOP 5 CLOSEST
    # ==========================================
    
    # STEP A: Calculate exact distance from every building to Little Montreal
    residential_gdf['distance_from_little_montreal'] = residential_gdf.geometry.distance(target_point_utm)
    
    # STEP B: Sort by distance (ascending) and strictly take the top 5
    # (We removed the radius filter to guarantee we get exactly 5 buildings)
    top_5_closest = residential_gdf.sort_values('distance_from_little_montreal', ascending=True).head(200)
        
    # ==========================================
    
    print(f"Found the top 5 closest residential buildings.")
    
    # 6. Convert filtered buildings back to Lat/Lon for the output
    nearby_buildings_wgs84 = top_5_closest.to_crs(epsg=4326)
    centroids = nearby_buildings_wgs84.geometry.centroid
    
    # 7. Map to your requested output format
    output_df = pd.DataFrame()
    output_df['latitude'] = centroids.y
    output_df['longitude'] = centroids.x
    output_df['name'] = nearby_buildings_wgs84.get('buildingid', '')
    output_df['heating_demand'] = nearby_buildings_wgs84.get('heat_kw', '')
    output_df['cooling_demand'] = ''
    output_df['waste_heat_source_type'] = ''
    output_df['floor_area'] = nearby_buildings_wgs84.get('effarea', '') 
    output_df['height'] = nearby_buildings_wgs84.get('storys', '') 
    output_df['fsa'] = ''
    output_df['area'] = nearby_buildings_wgs84.get('shapearea', '')
    output_df['type'] = 'residential'
    output_df['load_type'] = ''
    
    # Add the distance column to the final output so you can verify how close they are
    output_df['distance_meters'] = top_5_closest['distance_from_little_montreal'].values
    
    output_df.to_csv(output_csv, index=False)
    print(f"Successfully saved formatted data to {output_csv}")

# ==========================================
# Run the Function
# ==========================================
if __name__ == "__main__":
    # Target coordinates for little Montreal
    TARGET_LAT = 43.3078282
    TARGET_LON = -79.8006747
    
    # File paths
    INPUT_FILE = r"C:\Users\orjik\source\Research\thermal-network-api\data\Husni\Burlington_NewModel_GoWest_MattModel.csv"
    OUTPUT_FILE = r"C:\Users\orjik\source\research\thermal-network-api\data\Buildinguz\top_200_closest_to_little_montreal.csv"
    
    # Execute 
    extract_top_5_closest_buildings(
        input_csv=INPUT_FILE, 
        output_csv=OUTPUT_FILE, 
        target_lat=TARGET_LAT, 
        target_lon=TARGET_LON
    )