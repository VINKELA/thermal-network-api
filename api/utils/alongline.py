import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString
import polyline

def extract_buildings_along_route(input_csv, output_csv, encoded_polyline, exclude_csv, buffer_km=0.5):
    print(f"Reading main data from {input_csv}...")
    
    # 1. Load data and sanitize columns to lowercase
    df = pd.read_csv(input_csv, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lower()
    
    # ==========================================
    # NEW: EXCLUSION LOGIC
    # ==========================================
    print(f"Reading exclusion list from {exclude_csv}...")
    df_exclude = pd.read_csv(exclude_csv, sep=None, engine='python')
    df_exclude.columns = df_exclude.columns.str.strip().str.lower()
    
    # Determine the ID column in the exclusion file
    if 'buildingid' in df_exclude.columns:
        exclude_ids = df_exclude['buildingid'].astype(str).tolist()
    elif 'name' in df_exclude.columns:
        exclude_ids = df_exclude['name'].astype(str).tolist()
    else:
        exclude_ids = df_exclude.iloc[:, 0].astype(str).tolist()
        
    # Filter the main dataframe to remove these IDs
    if 'buildingid' in df.columns:
        df['match_id'] = df['buildingid'].astype(str)
        df = df[~df['match_id'].isin(exclude_ids)].copy()
        print(f"Excluded IDs found in {exclude_csv}. Remaining buildings to process: {len(df)}")
    else:
        raise KeyError("The main dataset does not contain a 'buildingid' column to match against.")
    # ==========================================

    # 2. Convert WKT strings to Shapely geometry objects
    print("Processing geometries (this may take a moment)...")
    df['geometry'] = df['geometry'].apply(wkt.loads)
    
    # 3. Create GeoDataFrame and set CRS (UTM Zone 17N)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=32617, inplace=True, allow_override=True)
    
    # 4. Filter for RESIDENTIAL buildings
    if 'buildingty' in gdf.columns:
        residential_gdf = gdf[gdf['buildingty'].str.upper() == 'RESIDENTIAL'].copy()
    else:
        residential_gdf = gdf.copy()

    # 5. Decode polyline and create a Shapely LineString
    decoded_coords = polyline.decode(encoded_polyline)
    lon_lat_coords = [(lon, lat) for lat, lon in decoded_coords]
    
    # Create GeoSeries for the route in WGS84, then reproject to UTM (Meters)
    route_wgs84 = gpd.GeoSeries([LineString(lon_lat_coords)], crs="EPSG:4326")
    route_utm = route_wgs84.to_crs(epsg=32617).iloc[0]
    
    # ==========================================
    # CORE LOGIC: BUILDINGS WITHIN PARAMETERIZED DISTANCE
    # ==========================================
    
    # Calculate exact distance from every building to the polyline route
    residential_gdf['distance_from_route'] = residential_gdf.geometry.distance(route_utm)
    
    # Convert buffer parameter to meters (e.g., 0.5km = 500 meters)
    buffer_meters = buffer_km * 1000
    
    # Filter for buildings within the buffer and sort by proximity
    buildings_in_buffer = residential_gdf[residential_gdf['distance_from_route'] <= buffer_meters].copy()
    buildings_in_buffer = buildings_in_buffer.sort_values('distance_from_route', ascending=True)
        
    # ==========================================
    
    num_found = len(buildings_in_buffer)
    print(f"Found {num_found} un-excluded residential buildings within {buffer_km}km of the route.")
    
    if num_found == 0:
        print("No eligible buildings found in this radius. Exiting without creating an output file.")
        return

    # 6. Convert filtered buildings back to Lat/Lon for the output
    nearby_buildings_wgs84 = buildings_in_buffer.to_crs(epsg=4326)
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
    
    # Add the distance column to the final output
    output_df['distance_meters'] = buildings_in_buffer['distance_from_route'].values
    
    output_df.to_csv(output_csv, index=False)
    print(f"Successfully saved formatted data to {output_csv}")

# ==========================================
# Run the Function
# ==========================================
if __name__ == "__main__":
    # Note the 'r' prefix before the string to handle backslashes properly in Python
    ENCODED_POLYLINE_1 = r"awggGlpzfNJoAVsAHeALqDToAX{@Zu@Uo@k@uCm@yD]{B_@uB{@aE}@}BgBgEcAsB_BoCwAaCsAcCY[e@]y@]a@[s@w@oB_CuAmB[c@e@}@y@sB_BqFu@gBm@iAoAcCg@aCSgASkBQ{AGoBBgAJy@Vk@b@_@dGmAzVmF~MuCl@KfAUIm@W}AKm@M]UY]S_Ae@m@[eAs@uAeAsA{@w@i@gBaA}@i@{Au@uCs@oAYDe@Lq@\e@v@o@Xa@bBoCrDcG`FuH|BmDp@cANi@JcAGw@OuADcAB_@To@~@mB|G{LjBiDjAuBd@cAzAiCXk@hAyAZ]nAeAJGN[Zw@LaACu@NeBVoAh@qEr@_Fr@wEfB}IJo@vAsHd@aCt@aELo@T}AVsBLo@F_AF}B@_DD_@I_A_@{E]aCaAwDQm@i@wAg@qAE]y@qBm@wAu@aBuAeCmCkD_CaCe@[_@_@aAaAoAyAqAyAoA}AiCyCwA}AuBaCU]NVcAoAsAgBoAkB{AuBMST_@{@kAg@o@mBoCcBeCq@w@cBgC_BeCq@aAqAaBiAaBqAsBYa@{@iAa@q@w@eAkDgF]a@w@kAM]yAoBiBmCiA}AuI{LqAiBwAuByCiEcCkDmDeFmFyHkD_F_@k@Sc@m@{@]_@cAwAm@}@k@cAg@k@i@aAm@w@cA{AwBaD_@k@}@mA]o@sGwJeAeBKQy@aA_AwA_AwAy@qAUm@qAsB{@gAy@kA{@mAwDsFm@}@Yc@u@}@w@_AuA_BwCgCw@w@iCeDu@eAm@iAiAgC_@gAs@sCC[c@gBYsAWoA_BmHQkAO_AI?Kk@G]k@aCi@eB{@aBm@{@{AaBgB_B{BuBwIqHiAcA}A_B_@i@aDyDw@aAOCa@e@MO]]ESsAuAWIWUeC_Ci@k@_BqBiBmCSg@eBcCUMOWiAiBe@}@[q@Yo@yCmHgAwBy@mAMOOm@e@q@s@_A]k@Ua@UK]i@IIqBoCsGqJyB_DmCcE]c@QYU^KCOSi@u@yAgBUU]Qu@o@_@Q{AiA}@_A[o@QWw@uAuAgCeBaDm@cAgDwGqB}Dk@gBe@kCe@cDq@oD}@_CaE_Iy@aBiCoEuCuD{AcBgAqAsC}BwC{BgAeAkAsA]Ke@k@YYu@{@sA}ASUC[a@a@]a@UIeBkBM[a@a@mAmA[O[Uy@{@AYQBII_@_@mCyCm@w@sBwCeA_BW_@Gc@O@GKe@q@aAyASs@cBcC}BmDkI}LeDaFsAhBI@IGe@_A"
    
    ENCODED_POLYLINE = r"eekgGzjafNP_A|@A`@AAoB|@?J?bA@b@CpCDrB@zCGtCMXIf@EhAGf@GrHeApB]`@GhB_@BFFHP@`@GFl@"
    # Define your parameterized buffer distance in kilometers
    SEARCH_RADIUS_KM = 1 
    
    # File paths
    INPUT_FILE = r"C:\Users\orjik\source\Research\thermal-network-api\data\Husni\Burlington_NewModel_GoWest_MattModel.csv"
    
    # The file containing the buildings you want to EXCLUDE
    #EXCLUDE_FILE_1 = r"C:\Users\orjik\source\research\thermal-network-api\data\Buildinguz\top_200_closest_to_little_montreal.csv" 
    EXCLUDE_FILE = r"C:\Users\orjik\source\research\thermal-network-api\data\Buildinguz\newdev.csv"

    # Where to save the new results
    #OUTPUT_FILE_1 = r"C:\Users\orjik\source\research\thermal-network-api\data\Buildinguz\buildings_along_route22.csv"
    OUTPUT_FILE = r"C:\Users\orjik\source\research\thermal-network-api\data\Buildinguz\buildings_along_route_waste_water.csv"

    # Execute 
    extract_buildings_along_route(
        input_csv=INPUT_FILE, 
        output_csv=OUTPUT_FILE, 
        encoded_polyline=ENCODED_POLYLINE,
        exclude_csv=EXCLUDE_FILE,
        buffer_km=SEARCH_RADIUS_KM
    )