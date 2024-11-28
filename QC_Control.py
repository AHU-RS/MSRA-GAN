#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script processes MODIS LST quality control (QC) files, extracts the ones that meet the quality requirements,
and saves them as TIFF files. The LST error flag must be <= 1k.

@version: Anaconda
@author: LeYongkang
@contact: 1363989042@qq.com
@software: PyCharm
@file: 12_CombineLst_PredictedAndModis.py
@time: 2020/2/5 11:10 PM
"""

import numpy as np
from osgeo import gdal
import os
import pandas as pd


# Function to open a TIF file and return data, width, height, geo transform, and projection
def opentif(filepath):
    dataset = gdal.Open(filepath)
    im_width = dataset.RasterXSize  # Number of columns in the raster
    im_height = dataset.RasterYSize  # Number of rows in the raster
    data = dataset.ReadAsArray(0, 0, im_width, im_height)  # Read raster data
    im_data = np.array(data)
    print("Shape of the opened TIF: ", im_data.shape)

    im_geotrans = dataset.GetGeoTransform()  # Geospatial transform info
    im_proj = dataset.GetProjection()  # Projection info
    return im_data, im_width, im_height, im_geotrans, im_proj


# Function to save data as a TIF file with the specified geospatial parameters
def savetif(dataset, path, im_width, im_height, im_geotrans, im_proj):
    print("Saving to:", path)
    driver = gdal.GetDriverByName("GTiff")
    outdataset = driver.Create(path, im_width, im_height, 1, gdal.GDT_Float32)
    outdataset.SetGeoTransform(im_geotrans)  # Set geospatial transform
    outdataset.SetProjection(im_proj)  # Set projection
    outdataset.GetRasterBand(1).WriteArray(dataset)  # Write data
    outdataset.GetRasterBand(1).SetNoDataValue(0)  # Set NoData value to 0
    print("File saved successfully.")


# Main processing block
if __name__ == "__main__":
    inDir = r"The mixed path for QC control files and files to be processed (placed in the same folder)."  # Input directory containing QC files
    Out_Dir = r"The output path for files processed through QC control."  # Output directory for processed LST files

    # Get a list of all QC files in the input directory
    InList_Qc = [infile for infile in os.listdir(inDir) if infile.endswith("_QC.tif")]

    # Loop through all QC files and process them
    for InFile in InList_Qc:

        # Skip if the output file already exists
        if not os.path.exists(Out_Dir + os.sep + "The file suffix added to the output files (ending with .tif)."):

            # Get the full path of the QC file
            in_Full_Dir = inDir + os.sep + InFile

            # Open QC TIF file and extract data
            InData = opentif(in_Full_Dir)
            in_Array = np.array(InData[0], dtype=np.uint8)

            # Convert decimal values to binary representation
            binary_repr_v = np.vectorize(np.binary_repr)
            new = binary_repr_v(in_Array, 8)

            # Quality control check: LST error flag is 00 (LST error flag <= 1k)
            # The 6-7th bits of the binary representation indicate LST error flag
            Error_mask = np.char.count(new, '00', start=0, end=2) == 1

            # Get the corresponding LST file (remove "_QC" and replace with "_Day")
            in_Full_Dir_Lst = in_Full_Dir[:-7] + "_Day.tif"
            Lst_Array = opentif(in_Full_Dir_Lst)[0]

            # Apply the error mask to the LST data, setting invalid values to 0
            Out_Lst_Array = np.where(Error_mask, Lst_Array, 0)
            print(Out_Lst_Array)

            # Save the processed LST data if the file doesn't already exist
            output_file_path = Out_Dir + os.sep + in_Full_Dir_Lst.split("\\")[-1]
            if not os.path.exists(output_file_path):
                print("Saving:", output_file_path)
                savetif(Out_Lst_Array, output_file_path, InData[1], InData[2], InData[3], InData[4])
