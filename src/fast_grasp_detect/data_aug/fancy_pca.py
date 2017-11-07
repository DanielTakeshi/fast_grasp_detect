"""
    Implementation of Fancy PCA data augmentation technique for images
    proposed by Krizhevsky et al. (2012).

    Author: David Wang
"""

import cv2
import numpy as np
from numpy import linalg as LA

def compute_covariance_matrix(dataset):
    """
    Computes the 3x3 covariance matrix of the set of all 3-dimensional RGB 
    vectors in the dataset.
    Params:
        dataset: List of NumPy arrays, each one corresponding to an RGB imag
            in the dataset. Each NumPy should be of dimension m x n x 3.
    Returns:
        The 3x3 covariance matrix of the RGB vectors in the dataset.
    """
    rgb_vectors = np.zeros(shape=(0, 3))

    for img in dataset:
        # Retrieve dimensions of the image.
        m, n, c = img.shape
        # Build an flattened array of 3-dimensional RGB vectors.
        flattened_arr = img.reshape((m * n, 3))
        rgb_vectors = np.concatenate((rgb_vectors, flattened_arr))

    # Subtract the mean RGB vector before performing PCA.
    mean = np.mean(rgb_vectors)
    rgb_vectors -= mean

    # Compute the covariance matrix.
    cov = np.cov(rgb_vectors.T)

    return cov

def fancy_pca(img, cov, sigma=0.1):
    """
    Performs Fancy PCA data augmentation on a single image.
    Params:
        img: NumPy array of the original image, should be of dimensions m x n x 3.
        cov: Covariance matrix of all 3-dimensional RGB vectors in the training
            dataset.
        sigma: The standard deviation of the Gaussian to use when sampling RGB
            directions to add to the original image.
    Returns:
        An augmented image of the same dimension as the input image.
    """
    # Compute the eigendecomposition of the covariance matrix (should be very fast
    # since it's only 3x3, but can be moved to augment_data function for minor
    # optimization).
    eig_vals, eig_vecs = LA.eig(cov)

    # Random weights sampled from a Normal distribution.
    alphas = np.random.normal(scale=sigma, size=3)
    # Calculate the offset to apply to each pixel.
    offset = np.dot(eig_vecs, alphas * eig_vals)

    # Apply the offset.
    m, n, c = img.shape
    flattened_arr = img.reshape((m * n, c))
    flattened_arr = flattened_arr + offset
    augmented_image = flattened_arr.reshape((m, n, c))
    augmented_image = np.clip(augmented_image, 0, 255)

    return augmented_image


if __name__ == '__main__':
    img_paths = ["rollout_0_grasp_0.png", "rollout_0_grasp_2.png"]

    imgs = []
    for img_path in img_paths:
        imgs.append(cv2.imread(img_path, 1))

    cov = compute_covariance_matrix(imgs)
    fancy_pca(imgs[0], cov, sigma=0.02)
