# Random image
import pylab as plt
import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from collections import namedtuple
from tqdm.auto import tqdm as tq
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

STEP = 2
VOCAB_SIZE = 1000
SVM_LAMDA = 10
SVM_ITER = 2000

def dense_sift(img, step=STEP):
    keypoints = []
    for i in range(step//2, img.shape[0], step):
        for j in range(step//2, img.shape[1], step):
            keypoints.append(cv2.KeyPoint(j, i, step))
    
    sift = cv2.SIFT_create()
    return sift.compute(img, keypoints)

def dense_sift_batch(batch, step=STEP):
    labels = []
    paths = []
    keypoints = []
    descriptors = []
    for cls, path, img in tq(batch):
        labels.append(cls)
        paths.append(path)
        kps, desc = dense_sift(img, step)
        keypoints.append(kps)
        descriptors.append(desc)
    return labels, paths, keypoints, descriptors

def bin_histograms(labels, descriptors, k):
    
    histograms = np.zeros((len(descriptors), k), dtype=np.float64)
    idx = 0
    for i, desc in enumerate(descriptors):
        for _ in range(desc.shape[0]):
            histograms[i, labels[idx]] += 1
            idx += 1
    return histograms

def cluster(descriptors, k=VOCAB_SIZE, kmeans=None):
    descriptors_flat = np.concatenate(descriptors)
    
    if kmeans is None:
        kmeans = KMeans(n_clusters=k, n_init=5)
        cluster_labels = kmeans.fit_predict(descriptors_flat)
    else:
        cluster_labels = kmeans.predict(descriptors_flat)
        k = kmeans.cluster_centers_.shape[0]
    
    histograms = bin_histograms(cluster_labels, descriptors, k)
            
    return kmeans, histograms


LEVEL_WEIGHTS = [1, 1/3]
SIZE = (148, 148)

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def histograms_pyramid(keypoints, descriptors, cluster_labels, image_shape=SIZE, level_weights=LEVEL_WEIGHTS, k=VOCAB_SIZE):
    keypoint_positions = np.full(image_shape[:2], -1)
    for i, kp in enumerate(keypoints[0]):
        keypoint_positions[int(kp.pt[0]), int(kp.pt[1])] = i
    
    histograms = np.empty((len(descriptors), 0), dtype=np.float64)
    labels_list = []
    idx = 0
    for i, d in enumerate(descriptors):
        labels_list.append(cluster_labels[idx:idx+d.shape[0]])
        idx += d.shape[0]
    
    for i, w in enumerate(level_weights):
        sz_r = image_shape[0] // (2**i)
        sz_c = image_shape[1] // (2**i)
        blocks = blockshaped(keypoint_positions, sz_r, sz_c)
        blocks = [np.ravel(blocks[i, :, :]) for i in range(blocks.shape[0])]
        blocks = [b[b != -1] for b in blocks]
        for b in blocks:
            sel_desc = [d[b, :] for d in descriptors]
            sel_lab = [l[b] for l in labels_list]
            
            hist = w * bin_histograms(np.concatenate(sel_lab), sel_desc, k)
            histograms = np.hstack((histograms, hist))
            
    return histograms    

def cluster_pyramid(keypoints, descriptors, image_shape=SIZE, k=VOCAB_SIZE, level_weights=LEVEL_WEIGHTS, kmeans=None):
    descriptors_flat = np.concatenate(descriptors)
    
    if kmeans is None:
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(descriptors_flat)
    else:
        cluster_labels = kmeans.predict(descriptors_flat)
        k = kmeans.cluster_centers_.shape[0]
    
    histograms = histograms_pyramid(keypoints, descriptors, cluster_labels, image_shape, level_weights, k)
            
    return kmeans, histograms


def load_train(imgs):
    for f in imgs:
        # if f.startswith('.'):
        #     continue
        # f_ = os.path.join(path, f)
        # img = cv2.imread(f_)
        # print("Box data {}".format(f))
        # print(type(f))
        img = cv2.resize(np.asarray(f), SIZE, interpolation=cv2.INTER_AREA)
        yield "pth", img
        
def load_train_all(lim=None):
    for path, img in load_train(lim):
        yield "cls", path, img

def extract_feature(lim=None):
    tr_labels, _, tr_keypoints, tr_descriptors = dense_sift_batch(lim)
    kmeans, tr_histograms = cluster(tr_descriptors)
    kmeans, tr_histograms = cluster_pyramid(tr_keypoints, tr_descriptors, kmeans=kmeans)
    return tr_histograms