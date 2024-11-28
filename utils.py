import os
import numpy as np
import h5py
import cv2 as cv
import random


def make_data(input1, input2, label):
    """Save the input data and labels to an HDF5 file."""
    savepath = 'Path to store the HDF5 file.'

    # Ensure the directory exists
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))

    # Write data to HDF5 file
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input1', data=input1)
        hf.create_dataset('input2', data=input2)
        hf.create_dataset('label', data=label)


def read_mask(mask_path):
    """Read and return all mask images from a given directory."""
    mask_img_sequence = []
    mask_file_path = get_filename(mask_path, '.tif')

    for mask_file in mask_file_path:
        mask_img = cv.imread(mask_file, cv.IMREAD_UNCHANGED)
        mask_img_sequence.append(mask_img)

    return mask_img_sequence


def get_filename(path, suffix):
    """Get all filenames with a specific suffix from a directory."""
    file_list = []
    f_list = os.listdir(path)
    f_list.sort(key=lambda x: str(x[:-4]))  # Sort filenames

    for file in f_list:
        if os.path.splitext(file)[1] == suffix:
            file_list.append(os.path.join(path, file))

    return file_list


def get_dir(path):
    """Get the folder names in a directory."""
    file_dir = []
    dirs = os.listdir(path)

    for dir in dirs:
        file_dir.append(os.path.join(path, dir))

    return file_dir


def input_prepare(dataset_path):
    """Prepare input data and labels for training."""
    image_size = 64

    # Paths to the input data directories
    t2_subori = r'Path to store the label image files.'
    t1_subdir = r'Path to store the reference image files.'
    t2_subdir = r'Path to store the simulated cloud cover mask image files.'

    sub_input1_sequence = []
    sub_input2_sequence = []
    sub_label_sequence = []

    # Get all filenames for the input data
    t1_dir = dataset_path + t1_subdir
    t2_dir = dataset_path + t2_subdir
    t2_ori_dir = dataset_path + t2_subori

    t1_names = get_filename(t1_dir, '.tif')
    t2_names = get_filename(t2_dir, '.tif')
    t2_oris = get_filename(t2_ori_dir, '.tif')

    total_images = len(t1_names)
    print(total_images)

    for num in range(total_images):
        t2_ori = t2_oris[num]
        t1_name = t1_names[num]
        t2_name = t2_names[num]

        # Read images
        t2_ori_img = cv.imread(t2_ori, cv.IMREAD_UNCHANGED)
        t1_img = cv.imread(t1_name, cv.IMREAD_UNCHANGED)
        t2_img = cv.imread(t2_name, cv.IMREAD_UNCHANGED)

        # Compute label as the difference between T2 original and T2 image
        label = t2_ori_img - t2_img

        # Reshape images to match the required size
        label = label.reshape([image_size, image_size, 1])
        t1_img = t1_img.reshape([image_size, image_size, 1])
        t2_img = t2_img.reshape([image_size, image_size, 1])

        sub_input1_sequence.append(t1_img)
        sub_input2_sequence.append(t2_img)
        sub_label_sequence.append(label)

    # Shuffle data
    input12_label = list(zip(sub_input1_sequence, sub_input2_sequence, sub_label_sequence))
    random.shuffle(input12_label)
    sub_input1_sequence[:], sub_input2_sequence[:], sub_label_sequence[:] = zip(*input12_label)

    # Convert to numpy arrays
    arrinput1 = np.asarray(sub_input1_sequence)
    arrinput2 = np.asarray(sub_input2_sequence)
    arrlabel = np.asarray(sub_label_sequence)

    # Save the prepared data
    make_data(arrinput1, arrinput2, arrlabel)


def read_data(path):
    """Read data from an HDF5 file."""
    with h5py.File(path, 'r') as hf:
        input1 = np.array(hf.get('input1'))
        input2 = np.array(hf.get('input2'))
        label = np.array(hf.get('label'))
        return input1, input2, label


# Test the script
if __name__ == '__main__':
    dataset_path = ''
    input_prepare(dataset_path)
