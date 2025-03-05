import pandas as pd
import rioxarray as rxr
import rasterio
from rasterio.mask import mask
from pyproj import Transformer, CRS
from shapely.geometry import Point, mapping
import numpy as np
import torch
from tqdm import tqdm

def extract_buffered_satellite_data(tiff_path, csv_path, buffer_distance=50, use_gpu=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")

    dataset = rxr.open_rasterio(tiff_path)
    tiff_crs = dataset.rio.crs
    print(f"Original TIFF CRS: {tiff_crs}")
    
    # Ensure the TIFF is in a projected coordinate system (not EPSG:4326)
    if not tiff_crs.is_projected:
        print("Reprojecting TIFF to UTM for correct buffering...")
        reprojected_tiff_path = "reprojected.tif"
        dataset = dataset.rio.reproject(dst_crs=CRS.from_epsg(32618))  # Example UTM Zone 18N
        dataset.rio.to_raster(reprojected_tiff_path)
        tiff_path = reprojected_tiff_path  # Use the reprojected file
        tiff_crs = dataset.rio.crs
        print(f"New TIFF CRS after reprojection: {tiff_crs}")
    
    df = pd.read_csv(csv_path)
    latitudes = df['Latitude'].values
    longitudes = df['Longitude'].values

    transformer = Transformer.from_crs("EPSG:4326", tiff_crs, always_xy=True)
    
    # Convert lat/lon to projected coordinates (meters)
    transformed_coords = np.array(transformer.transform(longitudes, latitudes))
    print("First 5 transformed coordinates (meters):")
    print(transformed_coords[:, :5])

    num_bands = dataset.shape[0]
    band_values = {f'B{band+1:02d}': np.full(len(df), np.nan, dtype=np.float32) for band in range(num_bands)}

    # Check raster bounds
    with rasterio.open(tiff_path) as src:
        raster_bounds = src.bounds
        print(f"Raster bounds: {raster_bounds}")
        
        for idx, (x, y) in tqdm(enumerate(zip(transformed_coords[0], transformed_coords[1])),
                                 total=len(latitudes), desc="Extracting values"):
            if not (raster_bounds.left <= x <= raster_bounds.right and raster_bounds.bottom <= y <= raster_bounds.top):
                print(f"Skipping point {idx} ({x:.2f}, {y:.2f}) - Outside raster bounds")
                continue
            
            point = Point(x, y).buffer(buffer_distance)
            geojson_geom = [mapping(point)]
            
            if idx < 5:  # Print debug info for the first few points
                print(f"Point {idx}: ({x:.2f}, {y:.2f}) -> Buffered Geometry: {geojson_geom}")
            
            try:
                out_image, _ = mask(src, geojson_geom, crop=True)
                
                # Compute the mean for each band
                for band in range(out_image.shape[0]):
                    band_values[f'B{band+1:02d}'][idx] = np.nanmean(out_image[band])
                
                if idx < 5:  # Debug output for the first few iterations
                    print(f"Extracted mean band values for point {idx}:", 
                          [np.nanmean(out_image[band]) for band in range(out_image.shape[0])])
            except Exception as e:
                print(f"Skipping point {idx} ({y:.2f}, {x:.2f}) due to error: {e}")

    band_df = pd.DataFrame(band_values)
    return band_df

# Example usage:
final_data = extract_buffered_satellite_data('S2_output_IC25(complete).tiff', 'Training_Data_IC25.csv')
# Save results
final_data.to_csv("output.csv", index=False)