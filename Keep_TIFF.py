import os

# Define the target folder path
folder_path = r'File storage path.'

# Traverse all files in the target folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)

        # Check if the file extension is .tif
        if file_path.endswith('.tif'):
            print(f"Retaining file: {file_path}")
        else:
            # If the file extension is not .tif, delete the file
            os.remove(file_path)  # Delete the file
            print(f"Deleted file: {file_path}")
