import pandas as pd
import numpy as np
import xlsxwriter as xw
import os
import math

# Define file paths
sta_path = 'Path to the output filtered station table data (ending with .xlsx).'
input_data_path = 'Path to the input station table data (ending with .xlsx).'

# Remove the existing output file if it exists
if os.path.exists(sta_path):
    os.remove(sta_path)

# Create a new Excel workbook and worksheet
workbook = xw.Workbook(sta_path)
worksheet1 = workbook.add_worksheet('sheet1')

# Read the input data
data = pd.read_excel(input_data_path)

# Initialize row index for Excel output
m1 = 0

# Loop through each day of the year (365 days)
n = 2  # Starting index for the dataset
for i in range(366):
    # Extract the relevant row for each day
    result = data.loc[[n]]
    result11 = result[['Timestamp', 'Rld', 'Rlu']]

    # Convert the result to a list for easier processing
    result22 = np.array(result11)
    result22 = result22.tolist()
    result33 = result22[0]

    # Extract values from the list
    date = str(result33[0])
    down = float(result33[1])  # Atmospheric downwelling radiation
    up = float(result33[2])  # Surface upwelling radiation

    # Check if both radiation values are valid (not NaN)
    if not (np.isnan(down)) and not (np.isnan(up)):
        # Calculate the temperature of the surface using the Stefan-Boltzmann law
        Lt = (up - (1 - 0.95) * down) / (5.67 * (10 ** -8) * 0.95)
        L = Lt ** 0.25  # Surface temperature (in Kelvin)

        # Ensure valid data (non-zero temperature and date in 2010)
        if date[0:4] == '2020' and L != 0:
            # Write the data to the Excel sheet
            worksheet1.write(m1, 0, date)  # Date/time
            worksheet1.write(m1, 1, result33[1])  # Atmospheric downwelling radiation
            worksheet1.write(m1, 2, result33[2])  # Surface upwelling radiation
            worksheet1.write(m1, 3, L)  # Surface temperature (Kelvin)
            m1 += 1  # Move to the next row in the Excel sheet

    # Increment to the next row in the dataset (24-hour interval)
    n += 24

# Close the workbook after processing
workbook.close()
