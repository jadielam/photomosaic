from random import randint
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.misc import imresize as resize
from scipy import spatial
from skimage.measure import block_reduce

def crop_and_resize(im, t_s):
    max_idx = np.argmax(im.shape[0:2])
    max_length = np.max(im.shape[0:2])
    min_length = np.min(im.shape[0:2])
    temp_shape = (min_length, min_length)
    start_pos = int(max_length - min_length / 2)
    if max_idx == 0:
        im = im[0 : min_length, :, :]
    else:
        im = im[:, start_pos : start_pos + min_length, :]
    return resize(im, (t_s, t_s))

def compute_statistics_on_images(source_images_path, t_s):
    images_paths = [y for x in os.walk(source_images_path) for y in glob(os.path.join(x[0], '*.jpg'))]
    images = []
    images_means = []
    images_stds = []
    
    for idx, im_path in enumerate(images_paths):
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype("float32")
        im = crop_and_resize(im, t_s)
        im_mean = np.mean(im, axis = (0, 1))
        im_std = np.std(im, axis = (0, 1))
        images.append(im)
        images_means.append(im_mean)
        images_stds.append(im_std)
        
    return images, images_means, images_stds

def transform_color(original_im, source_mean, source_std):
    '''
    Works better on lad color space
    '''
    target_im = original_im.copy()
    target_im_mean = np.mean(target_im, axis = (0, 1))
    target_im_std = np.std(target_im, axis = (0, 1))
    
    #Substract means from target image
    target_im[:,:,0] -= target_im_mean[0]
    target_im[:,:,1] -= target_im_mean[1]
    target_im[:,:,2] -= target_im_mean[2]
    
    #Scale by standard deviations
    #target_im[:,:,0] = (target_im_std[0] / source_std[0]) * target_im[:,:,0]
    #target_im[:,:,1] = (target_im_std[1] / source_std[1]) * target_im[:,:,1]
    #target_im[:,:,2] = (target_im_std[2] / source_std[2]) * target_im[:,:,2]
    
    #target_im[:,:,0] = (source_std[0] / target_im_std[0]) * target_im[:,:,0]
    #target_im[:,:,1] = (source_std[1] / target_im_std[1]) * target_im[:,:,1]
    #target_im[:,:,2] = (source_std[2] / target_im_std[2]) * target_im[:,:,2]
    
    #Add source mean
    target_im[:,:,0] += source_mean[0]
    target_im[:,:,1] += source_mean[1]
    target_im[:,:,2] += source_mean[2]
    
    # Clip pixel intensities to [0, 255] if they fall outside of the range
    target_im = np.clip(target_im, 0, 255)
    return target_im

def compute_corpus_statistics(source_images_path, tile_s):
    images, images_means, images_stds = compute_statistics_on_images(source_images_path, tile_s)
    tree = spatial.KDTree(np.array(images_means))
    return images, images_means, images_stds, tree
    
def create_tile_image(target_image_path, pooling, k, tile_s, images, images_means, images_stds, tree):
    '''
    Arguments:
    - target_image_path: (str)
    - pooling: (int)
    - k: (int)
        
    Returns:
    - tiled_image
    '''
    target_im = cv2.imread(target_image_path)
    target_im = cv2.cvtColor(target_im, cv2.COLOR_BGR2LAB).astype("float32")
    mean_reduced = block_reduce(target_im, block_size = (pooling, pooling, 1), func = np.mean)
    std_reduced = block_reduce(target_im, block_size = (pooling, pooling, 1), func = np.std)
        
    #1. Average pool of target image using h_c and w_c
    indexes_np = np.empty((mean_reduced.shape[0], mean_reduced.shape[1]), dtype=np.int32)
        
    #2. For each position on pool image, query KDTree to
    #find closest iamge not used and modify color.
    used_images = set()
    for i in range(indexes_np.shape[0]):
        for j in range(indexes_np.shape[1]):
            distances, indexes = tree.query(mean_reduced[i, j, :], k = k)
            u_i = 0
            while u_i < indexes.shape[0] and indexes[u_i] in used_images:
                u_i += 1
            if u_i >= indexes.shape[0]:
                random_index = randint(0, k - 1)
                index = indexes[random_index]
            else:
                index = indexes[u_i]
            used_images.add(index)                
            indexes_np[i, j] = index
                
    #3. return tiled image
    tiled_image = np.empty((tile_s * indexes_np.shape[0], tile_s * indexes_np.shape[1], 3), dtype = np.uint8)
    for i in range(indexes_np.shape[0]):
        for j in range(indexes_np.shape[1]):
            im = images[indexes_np[i, j]]
            im = transform_color(im.astype("float32"), mean_reduced[i, j,:], std_reduced[i, j, :])
            im = cv2.cvtColor(im.astype("uint8"), cv2.COLOR_LAB2BGR)
            tiled_image[i * tile_s : i * tile_s + tile_s, j * tile_s : j * tile_s + tile_s, :] = im
    #tiled_image = cv2.cvtColor(tiled_image.astype("uint8"), cv2.COLOR_LAB2BGR)
    return tiled_image
    
images, images_means, images_stds, tree = compute_corpus_statistics("./", 32 * 6)
resolution = 32
k = 100
tile_s = 32 * 6
tile_im = create_tile_image("./target_picture_1.jpg", resolution, k, tile_s, images, images_means, images_stds, tree)
