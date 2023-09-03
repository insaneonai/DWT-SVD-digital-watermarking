import time
import random
import cv2
import os
import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from math import sqrt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt

watermark_to_embed = np.load("yourwatermark.npy")
svds = np.linalg.svd(watermark_to_embed.reshape((32,32)))

# Detection, with these parameters, passes the 6 checks of the tester2.py script
def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0

    difference = img1 - img2
    same = not np.any(difference)
    if same is True:
        return 9999999
    csf = np.genfromtxt('utilities/csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels

def similarity(X, X_star):
    # Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
    return s

def extraction(input1, input2, input3):

    original_image = input1.copy()
    watermarked_image = input2.copy()
    attacked_image = input3.copy()

    alpha = 5.11  # 8 is the lower limit that can be used
    n_blocks_to_embed = 32
    block_size = 4
    watermark_size = 1024
    Uwm = svds[0].reshape((32,32))
    Vwm = svds[-1].reshape((32,32))

    # start time
    #start = time.time()

    blocks_with_watermark = []
    divisions = original_image.shape[0] / block_size
    watermark_extracted = np.float64(np.zeros(watermark_size))
    blank_image = np.float64(np.zeros((512, 512)))
    # compute difference between original and watermarked image

    difference = (watermarked_image - original_image)

    # fill blocks in differece where the difference is bigger o less than 0
    for i in range(0, original_image.shape[1], block_size):
        for j in range(0, original_image.shape[0], block_size):
            block_tmp = {'locations': (i, j)}
            if np.average(difference[i:i + block_size, j:j + block_size]) > 0:
                blank_image[i:i + block_size, j:j + block_size] = 1
                blocks_with_watermark.append(block_tmp)
            else:
                blank_image[i:i + block_size, j:j + block_size] = 0

    attacked_image-=np.uint8(blank_image)

####################################################################################################################


    shape_LL_tmp = np.floor(original_image.shape[0] / (2*divisions))
    shape_LL_tmp = np.uint8(shape_LL_tmp)

    watermark_extracted = np.zeros(1024).reshape(32, 32)
    Swm = np.zeros(32)
    #print(watermark_extracted)
    for i in range(len(blocks_with_watermark)):
        x = np.uint16(blocks_with_watermark[i]['locations'][0])
        y = np.uint16(blocks_with_watermark[i]['locations'][1])
        #get the block from the attacked image
        block = attacked_image[x:x + block_size, y:y + block_size]
        #compute the LL of the block
        Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
        LL_tmp = Coefficients[0]
        # SVD
        Uc, Sc, Vc = np.linalg.svd(LL_tmp)
        #get the block from the original image
        block_ori = original_image[x:x + block_size, y:y + block_size]
        #compute the LL of the block
        Coefficients_ori = pywt.wavedec2(block_ori, wavelet='haar', level=1)
        LL_ori = Coefficients_ori[0]
        # SVD
        Uc_ori, Sc_ori, Vc_ori = np.linalg.svd(LL_ori)

        Sdiff = Sc - Sc_ori

        Swm[(i*shape_LL_tmp)%watermark_extracted.shape[0]: (shape_LL_tmp+(i*shape_LL_tmp)%watermark_extracted.shape[0])] += abs(Sdiff/alpha)

    Swm /= watermark_extracted.shape[0]
    watermark_extracted = (Uwm).dot(np.diag(Swm)).dot(Vwm)
    watermark_extracted = watermark_extracted.reshape(1024)
    watermark_extracted /= np.max(watermark_extracted)


####################################################################################################################

    #end time
    #end = time.time()
    #print('[EXTRACTION] Time: %.2fs' % (end - start))

    #print(watermark_extracted)
    return watermark_extracted

def detection(input1, input2, input3):

    original_image = cv2.imread(input1, 0).copy()
    watermarked_image = cv2.imread(input2, 0).copy()
    attacked_image = cv2.imread(input3, 0).copy()

    # start time
    #start = time.time()

    alpha = 5.11
    n_blocks_to_embed = 32
    block_size = 4
    watermark_size = 1024
    T = 14.4

    Uwm = svds[0].reshape((32,32))
    Vwm = svds[-1].reshape((32,32))
    #extract watermark from watermarked image
    watermarked_image_dummy = watermarked_image.copy()
    watermark_extracted_wm = extraction(original_image, watermarked_image, watermarked_image_dummy)

    #starting extraction
    blocks_with_watermark = []
    divisions = original_image.shape[0] / block_size
    watermark_extracted = np.float64(np.zeros(watermark_size))
    blank_image = np.float64(np.zeros((512, 512)))
    # compute difference between original and watermarked image

    difference = (watermarked_image - original_image)

    # fill blocks in differece where the difference is bigger o less than 0
    for i in range(0, original_image.shape[1], block_size):
        for j in range(0, original_image.shape[0], block_size):
            block_tmp = {'locations': (i, j)}
            if np.average(difference[i:i + block_size, j:j + block_size]) > 0:
                blank_image[i:i + block_size, j:j + block_size] = 1
                blocks_with_watermark.append(block_tmp)
            else:
                blank_image[i:i + block_size, j:j + block_size] = 0

    attacked_image -= np.uint8(blank_image)


    shape_LL_tmp = np.floor(original_image.shape[0] / (2*divisions))
    shape_LL_tmp = np.uint8(shape_LL_tmp)


    watermark_extracted = np.zeros(1024).reshape(32, 32)
    Swm = np.zeros(32)

    # print(watermark_extracted)
    for i in range(len(blocks_with_watermark)):
        x = np.uint16(blocks_with_watermark[i]['locations'][0])
        y = np.uint16(blocks_with_watermark[i]['locations'][1])
        # get the block from the attacked image
        block = attacked_image[x:x + block_size, y:y + block_size]
        # compute the LL of the block
        Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
        LL_tmp = Coefficients[0]
        # SVD
        Uc, Sc, Vc = np.linalg.svd(LL_tmp)
        # get the block from the original image
        block_ori = original_image[x:x + block_size, y:y + block_size]
        # compute the LL of the block
        Coefficients_ori = pywt.wavedec2(block_ori, wavelet='haar', level=1)
        LL_ori = Coefficients_ori[0]
        # SVD
        Uc_ori, Sc_ori, Vc_ori = np.linalg.svd(LL_ori)

        Sdiff = Sc - Sc_ori

        Swm[(i * shape_LL_tmp) % watermark_extracted.shape[0]: (shape_LL_tmp + (i * shape_LL_tmp) % watermark_extracted.shape[0])] += abs(Sdiff / alpha)


    Swm /= watermark_extracted.shape[0]
    watermark_extracted = (Uwm).dot(np.diag(Swm)).dot(Vwm)
    watermark_extracted = watermark_extracted.reshape(1024)
    watermark_extracted /= np.max(watermark_extracted)


    #end of extraction

    sim = similarity(watermark_extracted_wm, watermark_extracted)
    if sim > T:
        watermark_status = 1
    else:
        watermark_status = 0

    output1 = watermark_status
    #output2 = wpsnr(watermarked_image, attacked_image)

    # end time
    #end = time.time()
    #print('[DETECTION] Time: %.2fs' % (end - start))

    return output1
