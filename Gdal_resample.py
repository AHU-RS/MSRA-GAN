import gdal
import gdalconst
import os
import shutil

# Define input and output file paths
outputfile_Path = r"Path to store the resampled image files."
inputfile_Path = r"Path to store the image files to be resampled."
referencefilefilePath = r"Path to store the benchmark reference image sample in .tif format (for geographic information retrieval)."

# Clean up the output directory if it exists
if os.path.exists(outputfile_Path):
    shutil.rmtree(outputfile_Path)
os.mkdir(outputfile_Path)

# List all files in the input directory
names = os.listdir(inputfile_Path)
file_num = len(names)

for i in range(file_num):
    image_name = names[i]
    inputfilePath = inputfile_Path + image_name
    outputfilePath = outputfile_Path + image_name

    # Open the input image file and the reference file
    inputrasfile = gdal.Open(inputfilePath, gdal.GA_ReadOnly)
    print(inputrasfile)
    outfile = gdal.Open(referencefilefilePath, gdal.GA_ReadOnly)

    # Get projection information of input and reference images
    inputProj = inputrasfile.GetProjection()
    outputProj = outfile.GetProjection()

    # Open the reference file to retrieve its properties
    referencefile = gdal.Open(referencefilefilePath, gdal.GA_ReadOnly)
    referencefileProj = referencefile.GetProjection()
    referencefileTrans = referencefile.GetGeoTransform()
    bandreferencefile = referencefile.GetRasterBand(1)

    # Get dimensions and bands of the reference image
    Width = referencefile.RasterXSize
    Height = referencefile.RasterYSize
    nbands = referencefile.RasterCount

    # Create the output file with the same properties as the reference image
    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(outputfilePath, Width, Height, nbands, gdal.GDT_Float32)  # Output data type is Float32
    output.SetGeoTransform(referencefileTrans)
    output.SetProjection(referencefileProj)

    # Reproject the input image to match the reference image projection and geotransform
    gdal.ReprojectImage(inputrasfile, output, outputProj, referencefileProj,
                        gdalconst.GRA_Bilinear, 0.0, 0.0)  # GRA_Bilinear for bilinear resampling

