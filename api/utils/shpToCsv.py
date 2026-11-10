import geopandas as gpd
import os

def convert_shp_to_csv(shp_filepath, csv_filepath):
    """
    Converts a shapefile (.shp) to a comma-separated values (.csv) file.
    
    Parameters:
    shp_filepath (str): The path to the input .shp file.
    csv_filepath (str): The path where the output .csv file will be saved.
    """
    
    # Check if the input file exists
    if not os.path.exists(shp_filepath):
        print(f"Error: The file {shp_filepath} was not found.")
        return

    try:
        print(f"Reading shapefile: {shp_filepath}...")
        # Read the shapefile into a GeoDataFrame
        gdf = gpd.read_file(shp_filepath)
        
        print("Converting to CSV...")
        # Export the GeoDataFrame to CSV
        # index=False prevents pandas from writing the row indices to the file
        gdf.to_csv(csv_filepath, index=False)
        
        print(f"Success! File saved to: {csv_filepath}")
        
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_file = r"C:\Users\orjik\source\Research\thermal-network-api\data\Burl2026_model with per building yearly loads\Burlington_NewModel_GoWest_MattModel.shp" # <--- CHANGE ME
    output_file = r"C:\Users\orjik\source\Research\thermal-network-api\data\Husni\Burlington_NewModel_GoWest_MattModel.csv"      # <--- CHANGE ME
    
    convert_shp_to_csv(input_file, output_file) 