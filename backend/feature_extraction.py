import cv2
import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from collections import Counter
import math


# --------------------------------------------------
# ENTROPY FUNCTION
# --------------------------------------------------

def entropy_calc(pixels):

    counts = Counter(pixels)
    total = len(pixels)

    ent = 0

    for c in counts.values():

        p = c / total
        ent -= p * math.log2(p)

    return ent


# --------------------------------------------------
# SYMMETRY FEATURES
# --------------------------------------------------

def symmetry_features(img):

    h_flip = np.fliplr(img)
    v_flip = np.flipud(img)

    d1_flip = np.transpose(img)
    d2_flip = np.fliplr(np.flipud(np.transpose(img)))

    return [

        np.mean(np.abs(img - h_flip)),
        np.mean(np.abs(img - v_flip)),
        np.mean(np.abs(img - d1_flip)),
        np.mean(np.abs(img - d2_flip)),

    ]


# --------------------------------------------------
# GLCM FUNCTION
# --------------------------------------------------

def glcm_features(gray):

    return graycomatrix(

        gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True

    )


# --------------------------------------------------
# MAIN FEATURE EXTRACTION FUNCTION
# --------------------------------------------------

def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image")


    # IMPORTANT: consistent resize across training + testing

    img = cv2.resize(img, (256, 256))


    # convert to grayscale (NO histogram equalization)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    pixels = gray.flatten()

    features = []


    # =====================================================
    # C FEATURES (Original grayscale)
    # =====================================================

    glcm = glcm_features(gray)

    features.extend([

        np.mean(pixels),
        np.std(pixels),
        skew(pixels),
        kurtosis(pixels),
        entropy_calc(pixels.tolist()),
        np.percentile(pixels, 25),
        np.median(pixels),
        np.percentile(pixels, 75),

        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],

        graycoprops(glcm, 'correlation')[0, 1],
        graycoprops(glcm, 'energy')[0, 1],

        graycoprops(glcm, 'contrast')[0, 2],
        graycoprops(glcm, 'correlation')[0, 2],
        graycoprops(glcm, 'energy')[0, 2],
        graycoprops(glcm, 'homogeneity')[0, 2],

        graycoprops(glcm, 'correlation')[0, 3],
        graycoprops(glcm, 'energy')[0, 3],

    ])


    # =====================================================
    # D FEATURES (Downsampled grayscale)
    # =====================================================

    down = cv2.resize(gray, (128, 128))

    d_pixels = down.flatten()

    glcm_d = glcm_features(down)

    features.extend([

        np.mean(d_pixels),
        np.std(d_pixels),
        entropy_calc(d_pixels.tolist()),
        np.percentile(d_pixels, 25),
        np.median(d_pixels),
        np.percentile(d_pixels, 75),

        graycoprops(glcm_d, 'correlation')[0, 0],
        graycoprops(glcm_d, 'energy')[0, 0],
        graycoprops(glcm_d, 'correlation')[0, 1],
        graycoprops(glcm_d, 'energy')[0, 1],
        graycoprops(glcm_d, 'correlation')[0, 2],
        graycoprops(glcm_d, 'energy')[0, 2],
        graycoprops(glcm_d, 'correlation')[0, 3],
        graycoprops(glcm_d, 'energy')[0, 3],

    ])


    # =====================================================
    # SYMMETRY FEATURES
    # =====================================================

    features.extend(symmetry_features(gray))


    # =====================================================
    # FILTERED FEATURES (Gaussian blur)
    # =====================================================

    filt = cv2.GaussianBlur(gray, (5, 5), 0)

    f_pixels = filt.flatten()

    glcm_f = glcm_features(filt)

    features.extend([

        skew(f_pixels),
        kurtosis(f_pixels),

        graycoprops(glcm_f, 'correlation')[0, 0],
        graycoprops(glcm_f, 'energy')[0, 0],
        graycoprops(glcm_f, 'correlation')[0, 1],
        graycoprops(glcm_f, 'energy')[0, 1],
        graycoprops(glcm_f, 'correlation')[0, 2],
        graycoprops(glcm_f, 'energy')[0, 2],
        graycoprops(glcm_f, 'correlation')[0, 3],
        graycoprops(glcm_f, 'energy')[0, 3],

    ])


    # =====================================================
    # WAVELET FEATURES
    # =====================================================

    coeffs = pywt.dwt2(gray, 'haar')

    LL, (LH, HL, HH) = coeffs


    def stats(x):

        return np.mean(x), np.std(x), skew(x.flatten()), kurtosis(x.flatten())


    _, W_RLHStd, _, W_RLHKurt = stats(LH)

    _, W_GLHStd, W_GLHSkew, W_GLHKurt = stats(HL)

    _, _, W_GHLSkew, W_GHLKurt = stats(HH)

    _, _, W_BLHSkew, W_BLHKurt = stats(LH)

    W_BHLMean, _, W_BHLSkew, _ = stats(HL)

    _, _, W_BHHSkew, _ = stats(HH)


    features.extend([

        W_RLHStd,
        W_RLHKurt,
        W_GLHStd,
        W_GLHSkew,
        W_GLHKurt,
        W_GHLSkew,
        W_GHLKurt,
        W_BLHSkew,
        W_BLHKurt,
        W_BHLMean,
        W_BHLSkew,
        W_BHHSkew

    ])


    return np.array(features).reshape(1, -1)