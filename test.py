import tensorflow as tf
import cv2 as cv
import os
import shutil
import xlsxwriter as xw
import numpy as np
from result_cmp import *
from utils import getfilename
import model  # Use the specific model


def test():
    # File paths
    file_name = 'Your file path'
    input1_path = 'Reference tif image file path'
    input2_path = 'Mask tif image file path'
    ori2_path = 'Label tif Image file path'

    # Mask paths
    mask_base_path = 'Masked sample image path (needs to be normalized to 0-255).'
    masks = [cv.imread(mask_base_path + f'{i}.tif', cv.IMREAD_UNCHANGED) for i in range(1, 9)]

    # Model path
    model_path = 'The  path of your model.'

    # Output paths
    out_path = 'The storage path for output results.'
    sta_path = 'The storage path for output-related tables.'

    # Image size and other configurations
    image_size = 64
    t1_names = os.listdir(input2_path)
    total_images = len(t1_names)

    # TensorFlow session and placeholders
    sess = tf.Session()
    x1 = tf.placeholder(tf.float32, [1, image_size, image_size, 1])
    x2 = tf.placeholder(tf.float32, [1, image_size, image_size, 1])

    # Load model
    pred = model.Generator(x1, x2, 3, False)
    checkpoint_dir = model_path
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.Saver()

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(f"Restoring model from {ckpt_name}")
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

    # Prepare output directory and Excel workbook
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if os.path.exists(sta_path):
        os.remove(sta_path)

    workbook = xw.Workbook(sta_path)
    worksheets = [workbook.add_worksheet(f'mask{i}') for i in range(1, 9)]

    # Initialize counters for each worksheet
    counters = [0] * 8

    # Process each image
    for i, t1_name in enumerate(t1_names):
        if t1_name[0] in '12345678':  # Mask selection based on file name
            mask_idx = int(t1_name[0]) - 1
            mask = masks[mask_idx]

            # Prepare file paths for images
            t1_path = os.path.join(input1_path, t1_name)
            t2_path = os.path.join(input2_path, t1_name)
            t2ori_path = os.path.join(ori2_path, t1_name)

            # Read input images
            input1 = cv.imread(t1_path, cv.IMREAD_UNCHANGED)
            input2 = cv.imread(t2_path, cv.IMREAD_UNCHANGED)
            ori2 = cv.imread(t2ori_path, cv.IMREAD_UNCHANGED)

            # Image size calculations
            sum_num = len(ori2) * len(ori2)

            # Preprocess mask and input images
            mask = mask / 255.0
            input1 = input1.reshape([1, image_size, image_size, 1])
            input2 = input2.reshape([1, image_size, image_size, 1])
            mask = mask.reshape([1, image_size, image_size, 1])

            # Run the model prediction
            tmp_result = sess.run([pred], feed_dict={x1: input1, x2: input2})
            tmp_result = np.array(tmp_result)
            result = tmp_result.squeeze()
            ori2 = ori2.squeeze()
            mask = mask.squeeze()

            # Combine the result with input2 and mask
            image = result * (1 - mask) + input2.squeeze()

            # Calculate residuals and performance metrics
            res = (ori2 - image) * 110
            res_nz = res[res.nonzero()]
            num_nz = len(res_nz)
            sqrtt = np.sqrt(np.mean(res_nz ** 2))
            R2 = performance_metric(ori2, image)

            # Update the corresponding worksheet
            worksheet = worksheets[mask_idx]
            worksheet.write(counters[mask_idx], 0, t1_name)
            worksheet.write(counters[mask_idx], 2, sqrtt)
            worksheet.write(counters[mask_idx], 4, R2)
            worksheet.write(counters[mask_idx], 6, num_nz / sum_num)

            counters[mask_idx] += 1

            # Save the output image
            out_pathtmp = os.path.join(out_path, t1_name)
            cv.imwrite(out_pathtmp, image)

    # Close the workbook
    workbook.close()
    print("Processing completed.")

if __name__ == '__main__':
   test()