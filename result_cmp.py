# Import necessary libraries
import math
import numpy as np
import tensorflow as tf
from skimage.measure import compare_ssim
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import convolve2d
import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants import pi


# 2D Gaussian filter function (same as MATLAB's fspecial('gaussian'))
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """Generate a 2D Gaussian mask."""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Convolution operation with a kernel
def filter2(x, kernel, mode='same'):
    """Apply a 2D filter to an image using convolution."""
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


# Compute Structural Similarity Index (SSIM)
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1.0):
    """Compute SSIM between two images."""
    if not im1.shape == im2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if len(im1.shape) > 2:
        raise ValueError("Please input images with 1 channel.")

    # Constants and pre-computations
    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(window)

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    # Compute means and variances
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    # Compute SSIM map and return average SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


# Root Mean Square Error (RMSE) calculation
def lst_evalu(result_img, ori_img, fac):
    """Compute RMSE between two images with scaling factor."""
    R, C = ori_img.shape
    rmse = 0.0
    for ri in range(R):
        for ci in range(C):
            dif2 = (result_img[ri][ci] - ori_img[ri][ci]) * fac
            rmse += dif2 * dif2
    rmse = rmse / (R * C)
    return math.sqrt(rmse)


# Quantitative evaluation including RMSE, SSIM, and CC (Correlation Coefficient)
def quantitative_evalu(result_img, ori_img):
    """Compute RMSE, SSIM, and CC between two images."""
    R, C = ori_img.shape
    rmse = 0.0
    for ri in range(R):
        for ci in range(C):
            rmse += (result_img[ri][ci] - ori_img[ri][ci]) ** 2
    rmse = sqrt(rmse / (R * C))

    # Compute SSIM
    ssim = compute_ssim(result_img, ori_img)

    # Compute Correlation Coefficient (CC)
    result_img_mean = np.mean(result_img)
    ori_img_mean = np.mean(ori_img)
    temp1 = temp2 = temp3 = 0.0
    for rk in range(R):
        for ck in range(C):
            temp1 += (result_img[rk][ck] - result_img_mean) * (ori_img[rk][ck] - ori_img_mean)
            temp2 += (result_img[rk][ck] - result_img_mean) ** 2
            temp3 += (ori_img[rk][ck] - ori_img_mean) ** 2
    cc = temp1 / math.sqrt(temp2 * temp3)

    return rmse, ssim, cc


# Peak Signal-to-Noise Ratio (PSNR) calculation
def compute_psnr(target, ref, max_valu=1.0):
    """Compute PSNR between two images."""
    target_data = np.array(target, dtype=np.float32)
    ref_data = np.array(ref, dtype=np.float32)
    diff = (ref_data - target_data) ** 2
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff))
    return 20 * math.log10(max_valu / rmse)


# Compute R-squared score
def performance_metric(ori, result):
    """Compute R-squared (R2) score for model performance."""
    return r2_score(ori, result)


# Spectral Angle Mapper (SAM) for evaluating similarity
def compute_SAM(origin, result):
    """Compute Spectral Angle Mapper (SAM) between two images."""
    shape = origin.shape
    X1, Y1 = [], []
    for i in range(shape[0]):
        for j in range(shape[0]):
            X1.append(origin[i, j])
            Y1.append(result[i, j])

    X = np.array(X1)
    A = X.reshape(shape[0] * shape[1], 1)
    AT = X.reshape(1, shape[0] * shape[1])
    Y = np.array(Y1)
    B = Y.reshape(shape[0] * shape[1], 1)
    BT = Y.reshape(1, shape[0] * shape[1])

    m = np.sqrt(np.dot(AT, A))
    n = np.sqrt(np.dot(BT, B))
    mn = np.dot(m, n)
    p1 = np.dot(AT, B) / mn
    p2 = float(p1)

    return 1 / np.cos(p2)


# SSIM computation using Gaussian kernel
def SSIM_Computing(img_mat_1, img_mat_2):
    """Compute SSIM using a Gaussian filter."""
    # Define Gaussian kernel parameters
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))

    # Generate Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) * \
                                    exp(-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    # Convert images to double precision
    img_mat_1 = img_mat_1.astype(np.float)
    img_mat_2 = img_mat_2.astype(np.float)

    # Square of input images
    img_mat_1_sq = img_mat_1 ** 2
    img_mat_2_sq = img_mat_2 ** 2
    img_mat_12 = img_mat_1 * img_mat_2

    # Apply Gaussian filtering
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)

    # Variance and covariance computations
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel) - img_mat_mu_1_sq
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel) - img_mat_mu_2_sq
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel) - img_mat_mu_12

    # SSIM constants
    c_1 = 6.5025
    c_2 = 58.5225

    # Numerator and denominator of SSIM
    num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
    den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)

    # SSIM result
    ssim_map = num_ssim / den_ssim
    return np.average(ssim_map)
