##This code combines the processed MODIS LST with the corresponding CLDAS LST for the given date, in order to return the real MODIS LST (batch processing).
import os
import numpy as np
import cv2
from osgeo import gdal

# Define paths
image_path = r'Path to the MODIS image file to be reconstructed.'  # Path for reconstructed images
cldas_path = r'Path to store the CLDAS image files.'  # Path for T2 CLDAS data
out_path = r'Path to the output of the real temperature values from MODIS LST.'  # Output path for results

# Create output folder if it does not exist
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Get list of subfolders in the image directory
subfolders = [f.name for f in os.scandir(image_path) if f.is_dir()]

# Loop through each subfolder
for folder in subfolders:
    out_folder_path = os.path.join(out_path, folder)

    # Skip if the output folder already exists
    if os.path.exists(out_folder_path):
        print(f"Folder {out_folder_path} already exists, skipping.")
        continue

    os.makedirs(out_folder_path)

    image_folder_path = os.path.join(image_path, folder)
    cldas_folder_path = os.path.join(cldas_path, folder)

    # Skip if corresponding CLDAS folder doesn't exist
    if not os.path.exists(cldas_folder_path):
        print(f"Corresponding folder {cldas_folder_path} not found, skipping.")
        continue

    # Get list of image files in the current folder
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.tif')]

    # Loop through each image file
    for image_name in image_files:
        image_file_path = os.path.join(image_folder_path, image_name)
        cldas_file_path = os.path.join(cldas_folder_path, image_name)

        # Skip if the corresponding CLDAS file doesn't exist
        if not os.path.exists(cldas_file_path):
            print(f"CLDAS file {cldas_file_path} not found, skipping.")
            continue

        # Read the image and CLDAS data using OpenCV
        image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        cldas = cv2.imread(cldas_file_path, cv2.IMREAD_UNCHANGED)

        # Process the CLDAS data
        cldas[np.isnan(cldas)] = 0  # Replace NaN values with 0
        cldas = (cldas - 273.15 + 50) / 110  # Convert to appropriate units

        # Calculate the result based on the formula
        result = image - 0.5 + cldas
        result = result * 110 - 50 + 273.15  # Convert back to Kelvin
        result[np.isnan(result)] = 0  # Ensure no NaN values
        # result[result < 200] = 0  # Mask values below 200
        # result[result > 400] = 0  # Mask values above 400

        # Get geospatial information from the image
        image_ds = gdal.Open(image_file_path)
        geo_transform = image_ds.GetGeoTransform()
        projection = image_ds.GetProjection()
        image_ds = None  # Close the dataset

        # Write the result to the output path using the same geospatial info
        out_file = os.path.join(out_folder_path, image_name)
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(out_file, result.shape[1], result.shape[0], 1, gdal.GDT_Float32)
        out_ds.SetProjection(projection)
        out_ds.SetGeoTransform(geo_transform)
        out_ds.GetRasterBand(1).WriteArray(result)
        out_ds = None  # Close the output dataset
