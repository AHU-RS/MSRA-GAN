import tensorflow as tf
import cv2 as cv
import os
import model  # Import the model containing the Generator
import numpy as np

def testreal():
    """
    This function processes image pairs using a pre-trained model, applies a mask,
    and saves the resulting images in the specified output directory.
    """

    # Define base directories for input images, masks, and output results
    base_t1_path = r'Path to the spatially complete reference image file.'
    base_t2_path = r'Path to the MODIS image file to be reconstructed.'
    base_mask_path = r'Path to the mask file for the MODIS image to be reconstructed.'
    base_out_path = r'Path for storing the reconstruction results.'

    # Get list of subdirectories (each corresponding to a specific set of images)
    t1_subdirs = [d for d in os.listdir(base_t1_path) if os.path.isdir(os.path.join(base_t1_path, d))]

    # Define model path and parameters
    model_path = 'Path for storing the model.'
    image_size = 64

    # Set up the TensorFlow session and model placeholders
    sess = tf.Session()
    x1 = tf.placeholder(tf.float32, [1, image_size, image_size, 1])  # Input image 1
    x2 = tf.placeholder(tf.float32, [1, image_size, image_size, 1])  # Input image 2
    pred = model.Generator(x1, x2, 3, False)  # Define the model generator

    # Load the model checkpoint
    checkpoint_dir = model_path
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.Saver()

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    else:
        print("No checkpoint found. Exiting.")
        return

    # Iterate over each subdirectory (representing a set of images)
    for subdir in t1_subdirs:
        # Define paths for input images, masks, and output directory
        t1_path = os.path.join(base_t1_path, subdir)
        t2_path = os.path.join(base_t2_path, subdir)
        mask_path = os.path.join(base_mask_path, subdir)
        out_path = os.path.join(base_out_path, subdir)

        # Check if the output directory already contains the expected number of files
        if os.path.exists(out_path):
            tif_files = [f for f in os.listdir(out_path) if f.endswith('.tif')]
            if len(tif_files) == 9776:
                print(f"Skipping {subdir} as output folder already contains 9776 tif files.")
                continue
        else:
            os.makedirs(out_path)

        # Get list of image files in the current subdirectory
        t1_names = os.listdir(t1_path)
        sum_files = len(t1_names)
        print(f"Processing {sum_files} files in {subdir}")

        # Process each image pair
        for image_name in t1_names:
            # Define paths for the current image and mask
            t1name_path = os.path.join(t1_path, image_name)
            t2name_path = os.path.join(t2_path, image_name)
            maskname_path = os.path.join(mask_path, image_name)
            outname_path = os.path.join(out_path, image_name)

            # Skip processing if the output file already exists
            if os.path.exists(outname_path):
                print(f"Skipping {image_name} as it already exists.")
                continue

            # Load images and mask
            input1 = cv.imread(t1name_path, cv.IMREAD_UNCHANGED)
            input2 = cv.imread(t2name_path, cv.IMREAD_UNCHANGED)
            mask = cv.imread(maskname_path, cv.IMREAD_UNCHANGED)

            # Check if any images or masks failed to load
            if input1 is None or input2 is None or mask is None:
                print(f"Error loading image at {t1name_path}, {t2name_path}, or {maskname_path}")
                continue

            # Normalize the mask (assumed binary values: 0 or 255)
            mask = mask / 255.0

            # Reshape images and mask to fit model input requirements
            input1 = input1.reshape([1, image_size, image_size, 1])
            input2 = input2.reshape([1, image_size, image_size, 1])
            mask = mask.reshape([1, image_size, image_size, 1])

            # Run the model to generate a result
            tmp_result = sess.run([pred], feed_dict={x1: input1, x2: input2})
            tmp_result = np.array(tmp_result)
            result = tmp_result.squeeze()  # Remove extra dimensions
            mask = mask.squeeze()
            input2 = input2.squeeze()

            # Apply the mask to the result (combine with the second input image)
            image = result * (1 - mask) + input2

            # Save the output image
            cv.imwrite(outname_path, image)

if __name__ == '__main__':
    testreal()
