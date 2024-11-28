import os
import numpy as np
import cv2
from osgeo import gdal

# Define input and output paths
modis_path = r'The storage path for MODIS images.'
cldas_path = r'The storage path for CLDAS  images.'
out_path = r'The storage path for output results.'

# Create output directory if it doesn't exist
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Get list of all .tif files in the MODIS path
img_path_list = [img for img in os.listdir(modis_path) if img.endswith('.tif')]

# Process each image
for image_name in img_path_list:
    # Read the MODIS and CLDAS images
    modis = cv2.imread(os.path.join(modis_path, image_name), cv2.IMREAD_UNCHANGED)
    cldas = cv2.imread(os.path.join(cldas_path, image_name), cv2.IMREAD_UNCHANGED)

    # Check if the images have the same shape
    if modis.shape != cldas.shape:
        print(f"Image shapes do not match for {image_name}")
        continue

    # Apply processing to the MODIS image
    modis[modis == 0] = np.nan  # Set zero values to NaN
    modis = (modis - 273.15 + 50) / 110  # Convert temperature from Kelvin to Celsius, then scale

    # Apply processing to the CLDAS image
    cldas[cldas == 0] = np.nan  # Set zero values to NaN
    cldas = (cldas - 273.15 + 50) / 110  # Convert temperature from Kelvin to Celsius, then scale

    # Calculate the difference (MODIS - CLDAS) and add a constant
    Sub = modis - cldas + 0.5
    Sub[np.isnan(Sub)] = 0  # Replace NaN values with 0

    # Get geotransform and projection from MODIS image
    dataset = gdal.Open(os.path.join(modis_path, image_name))
    if dataset is not None:
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        dataset = None  # Close the dataset

        # Create output image with the same geotransform and projection
        output_path = os.path.join(out_path, image_name)
        driver = gdal.GetDriverByName('GTiff')
        output_ds = driver.Create(output_path, Sub.shape[1], Sub.shape[0], 1, gdal.GDT_Float32)
        output_ds.SetGeoTransform(geotransform)
        output_ds.SetProjection(projection)
        output_ds.GetRasterBand(1).WriteArray(Sub)
        output_ds.FlushCache()
        output_ds = None  # Close the output dataset
