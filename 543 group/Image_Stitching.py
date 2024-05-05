#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import skimage
import skimage.io
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy


# In[ ]:


def imread(fname):
    """
    read image into np array from file
    """
    return skimage.io.imread(fname)

def imread_bw(fname):
    """
    read image as gray scale format
    """
    return cv2.cvtColor(imread(fname), cv2.COLOR_BGR2GRAY)

def imshow(img):
    """
    show image
    """
    skimage.io.imshow(img)
    
def get_sift_data(img):
    """
    detect the keypoints and compute their SIFT descriptors with opencv library
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    plot the match between two image according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')


# In[ ]:


def get_best_matches(img1, img2, num_matches):
    kp1, des1 = get_sift_data(img1)
    kp2, des2 = get_sift_data(img2)
    kp1, kp2 = np.array(kp1), np.array(kp2)
    
    # Find distance between descriptors in images
    dist = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')
    
    # Select putative matches that selects the pairs with the smallest distances
    idx_sort = np.argsort(dist, axis=None)[:num_matches]
    match_idx = np.unravel_index(idx_sort, dist.shape)
    
    # Extracting the coordinates of the matching keypoints
    p1 = np.array([kp1[idx].pt for idx in match_idx[0]])
    p2 = np.array([kp2[idx].pt for idx in match_idx[1]])
    
    inliers = np.hstack([p1, p2])
    return inliers

def compute_homography(sample_points):
    """
    code to compute homography according to the matches
    """
    A = []
    for points in sample_points:
        x1, y1, x2, y2 = points
        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])

    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H

def apply_homography(H, points):
    """
    Apply homography to points.
    """
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = points_homogeneous @ H.T
    transformed_points /= transformed_points[:, -1][:, None]
    return transformed_points[:, :2]

def ransac(matched_points, threshold, e):
    """
    ransac code to find the best model, inliers, and residuals
    """
    best_homography = None
    best_inliers = None
    best_residual = None
    max_inliers = 0
    
    N = int(np.log(1 - 0.99)/np.log(1 - (1 - e)**4))
    for i in range(N):
        # Randomly sample 4 matches for homography estimation
        sample_indices = np.random.choice(len(matched_points), 4, replace=False)
        sample_points = matched_points[sample_indices]

        # Compute homography matrix using the selected 4 matches
        H = compute_homography(sample_points)

        # Project source points to the destination using the computed homography
        projected_points = apply_homography(H, matched_points[:, :2])

        # Calculate residuals
        residuals = np.sum((matched_points[:, 2:] - projected_points) ** 2, axis=1)
        
        # Filter inliers and outliers according to threshold
        inliers = matched_points[residuals <= threshold]
        outliers = matched_points[residuals > threshold]
        num_inliers = len(inliers)
        
        # Calculate the mean residual for inliers
        current_residual = np.mean(residuals[residuals <= threshold])

        # Update the best model if the current one has more inliers
        if num_inliers > max_inliers:
            best_homography = H
            best_inliers = inliers
            best_residual = current_residual
            max_inliers = num_inliers

    return best_homography, best_inliers, best_residual
    
def warp_images(image1, image2, homography):
    """
    code to stitch images together according to the homography
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Transform corners of image1
    corners_image1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32)
    corners_image1_transformed = cv2.perspectiveTransform(np.array([corners_image1]), homography)[0]
    
    # Compute corners of the resulting image
    all_corners = np.concatenate((corners_image1_transformed, [[0, 0], [w2, 0], [w2, h2], [0, h2]]), axis=0)
    
    # Find the extents of the transformed image
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translate points to the new view
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], 
                              [0, 1, translation_dist[1]], 
                              [0, 0, 1]])
    
    # Warp image to the new panorama canvas
    output_image = cv2.warpPerspective(image1, H_translation.dot(homography), (x_max - x_min, y_max - y_min))
    output_image[translation_dist[1]:translation_dist[1] + h2, 
                 translation_dist[0]:translation_dist[0] + w2] = image2
    
    # Normalize the image
    output_image = np.uint8(output_image / output_image.max() * 255)

    return output_image


# Load images

# In[ ]:


img1 = imread('test_left.jpg')
img2 = imread('test_right.jpg')


# Compute and display the initial SIFT matching result

# In[ ]:


data = get_best_matches(img1, img2, 500)
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, img1, img2, data)
fig.savefig('sift_match.pdf', bbox_inches='tight')


# Perform RANSAC to get the homography and inliers

# In[ ]:


# display the inlier matching, report the average residual
# <YOUR CODE>
homography, max_inliers, best_model_errors = ransac(data, 30, 0.2)
print("Average residual:", np.average(best_model_errors))
print("Inliers:", max_inliers)
fig.savefig('ransac_match.png', bbox_inches='tight')
print(data.shape, max_inliers.shape)


# Warp images to stitch them together

# In[ ]:


# display and report the stitching results
# <YOUR CODE>
im = warp_images(img1, img2, homography)
cv2.imwrite('stitched_images.jpg', im[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])

