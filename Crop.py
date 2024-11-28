import os
import gc
import multiprocessing

try:
    import gdal
except:
    from osgeo import gdal
import numpy as np

# Function to read a TIFF dataset
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + " could not be opened")
    return dataset


# Function to write a NumPy array to a TIFF file
def writeTiff(im_data, im_geotrans, im_proj, path):
    # Determine the data type for the TIFF file
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # Handle both 2D and 3D image data
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])  # Convert to 3D array
        im_bands, im_height, im_width = im_data.shape

    # Create the TIFF file
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)

    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # Write geotransform
        dataset.SetProjection(im_proj)  # Write projection

    # Write each band of the image data to the TIFF file
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset  # Release the dataset from memory


# Function to transform pixel coordinates to geographic coordinates using affine transformation
def CoordTransf(Xpixel, Ypixel, GeoTransform):
    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    return XGeo, YGeo

# Function to crop a TIFF image into smaller patches
def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    # Create output folder if it doesn't exist
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)

    # Read the input image data
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # Read the full image

    # Initialize file name counter
    new_name = len(os.listdir(SavePath)) + 1

    # Crop the image into smaller patches with overlap based on RepetitionRate
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):

            # Skip if file already exists
            if os.path.isfile(SavePath + "/%d.tif" % new_name):
                print(f"File {new_name}.tif already exists, skipping")
                new_name += 1
                continue

            # Crop the image based on whether it's 2D or 3D (multiple bands)
            if len(img.shape) == 2:
                cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                              int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            else:
                cropped = img[:,
                              int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                              int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]

            # Calculate new geographic coordinates for the cropped area
            XGeo, YGeo = CoordTransf(int(j * CropSize * (1 - RepetitionRate)),
                                     int(i * CropSize * (1 - RepetitionRate)),
                                     geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

            # Save the cropped image to disk
            writeTiff(cropped, crop_geotrans, proj, SavePath + "/%d.tif" % new_name)
            new_name += 1

    # Handle cropping for right edge
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):

        # Crop right edge of the image
        if len(img.shape) == 2:
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize): width]
        else:
            cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize): width]

        # Calculate new geographic coordinates
        XGeo, YGeo = CoordTransf(width - CropSize,
                                 int(i * CropSize * (1 - RepetitionRate)),
                                 geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

        # Save cropped image
        writeTiff(cropped, crop_geotrans, proj, SavePath + "/%d.tif" % new_name)
        new_name += 1

    # Handle cropping for bottom edge
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):

        # Crop bottom edge of the image
        if len(img.shape) == 2:
            cropped = img[(height - CropSize): height,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                          (height - CropSize): height,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]

        # Calculate new geographic coordinates
        XGeo, YGeo = CoordTransf(int(j * CropSize * (1 - RepetitionRate)),
                                 height - CropSize,
                                 geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

        # Save cropped image
        writeTiff(cropped, crop_geotrans, proj, SavePath + "/%d.tif" % new_name)
        new_name += 1

    # Handle cropping for bottom-right corner
    if len(img.shape) == 2:
        cropped = img[(height - CropSize): height, (width - CropSize): width]
    else:
        cropped = img[:, (height - CropSize): height, (width - CropSize): width]

    # Calculate geographic coordinates for the bottom-right corner
    XGeo, YGeo = CoordTransf(width - CropSize, height - CropSize, geotrans)
    crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

    # Save cropped image
    writeTiff(cropped, crop_geotrans, proj, SavePath + "/%d.tif" % new_name)

# Function to process a single file
def process_file(file_path, output_path, CropSize, RepetitionRate):
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists, skipping this file")
        return
    TifCrop(file_path, output_path, CropSize, RepetitionRate)
    gc.collect()

if __name__ == '__main__':
    # Input and output folders, and crop parameters
    input_folder = r"Path to the large image to be cropped."
    output_folder = r"Path to store the cropped small images."
    CropSize = 64
    RepetitionRate = 0.5

    # Create list of tasks (file paths to be processed)
    imglist2 = os.listdir(input_folder)
    tasks = []

    for j in range(0, 366):  # Assuming processing for 366 files (e.g., days in a leap year)
        name2_tuple = os.path.splitext(imglist2[j])[0]
        name2_str = ''.join(name2_tuple)
        file_path = os.path.join(input_folder, imglist2[j])
        output_path = os.path.join(output_folder, name2_str[0:8])  # Use the first 8 characters of the filename for the folder
        tasks.append((file_path, output_path, CropSize, RepetitionRate))

    # Process the files in parallel using multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_file, tasks)
