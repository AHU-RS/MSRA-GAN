import tensorflow as tf
import cv2 as cv
import os
import shutil
import numpy as np
import model


def testreal():
    """
    This function processes image pairs (t1, t2) using a trained model, applies a mask,
    and reconstructs the result in the specified output directory.
    """

    # Define paths for the input images, masks, and output results
    t1_path = r'Path to the spatially complete reference image file.'
    t2_path = r'Path to the MODIS image file to be reconstructed.'
    mask_path = r'Path to the mask file for the MODIS image to be reconstructed.'
    out_path = r'Path for storing the reconstruction results.'

    # Create output directory if it doesn't exist, or clear it if it does
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    # List all image files in the mask directory
    t1_names = os.listdir(mask_path)
    total_images = len(t1_names)
    print(f"Total images to process: {total_images}")

    # Model directory and checkpoint
    model_path = 'Path for storing the model.'

    # Image size for processing
    image_size = 64

    # TensorFlow session setup
    sess = tf.Session()
    x1 = tf.placeholder(tf.float32, [1, image_size, image_size, 1])  # Placeholder for first input image
    x2 = tf.placeholder(tf.float32, [1, image_size, image_size, 1])  # Placeholder for second input image

    # Define the model generator (from model_2024_0411)
    pred = model.Generator(x1, x2, 3, False)

    # Load model checkpoint
    checkpoint_dir = model_path
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.Saver()

    # Iterate over all images and process each
    for i in range(total_images):
        image_name = t1_names[i]
        print(f"Processing image {i + 1}/{total_images}: {image_name}")

        # Define the full paths for the current image pair and mask
        t1name_path = t1_path + image_name
        t2name_path = t2_path + image_name
        maskname_path = mask_path + image_name

        # Restore model weights from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

            # Read input images and mask
            input1 = cv.imread(t1name_path, cv.IMREAD_UNCHANGED)
            input2 = cv.imread(t2name_path, cv.IMREAD_UNCHANGED)
            mask = cv.imread(maskname_path, cv.IMREAD_UNCHANGED)

            # Check if any of the images or mask failed to load
            if input1 is None or input2 is None or mask is None:
                print(f"Error loading images at {t1name_path}, {t2name_path}, or {maskname_path}")
                continue  # Skip this image pair

            # Normalize the mask (assumed to be binary: 0 or 255)
            mask = mask / 255.0

            # Reshape inputs and mask for TensorFlow
            input1 = input1.reshape([1, image_size, image_size, 1])
            input2 = input2.reshape([1, image_size, image_size, 1])
            mask = mask.reshape([1, image_size, image_size, 1])

            # Run the model to get the predicted result
            tmp_result = sess.run([pred], feed_dict={x1: input1, x2: input2})
            tmp_result = np.array(tmp_result)
            result = tmp_result.squeeze()  # Remove single-dimensional entries
            mask = mask.squeeze()
            input2 = input2.squeeze()

            # Combine the predicted result with the original image using the mask
            image = result * (1 - mask) + input2

            # Save the output image to the specified directory
            outname_path = out_path + image_name
            cv.imwrite(outname_path, image)
            print(f"Saved result for {image_name} at {outname_path}")


if __name__ == '__main__':
    testreal()
