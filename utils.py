import os
import xml.etree.ElementTree as ET
import cv2
import math
import numpy as np
from sklearn.cluster import KMeans
from skimage import io
from sklearn import preprocessing
from Saliency_Filter import *
from ncut import *
from collections import defaultdict
from scipy import stats
from skimage import io
from numpy.linalg import inv

"""
    function caculate is the discriminative clustering term 
    CM: central projection matrix
    I: ones matrix
    X: similarity natrix (n x d)
"""

def discriminative_optimial(central_matrix, X, nbox, I, k):
    # print("X shape .{}".format(X.shape))
    # print("I shape .{}".format(I.shape))
    # print("CM shape .{}".format(central_matrix.shape))
    _in_ = X.T @ central_matrix @ X + nbox * k
    I1 = np.identity(X.shape[0])
    _in1_ = I1 - X @ inv(np.matrix(_in_)) @ X.T
    A = (1/nbox) * central_matrix @ _in1_ @ central_matrix 
    return A

def normalize_laplacian(W, nbEigenValues):
    # parameters
    offset = .5
    maxiterations = 100
    eigsErrorTolerence = 1e-6
    truncMin = 1e-6
    eps = 2.2204e-16

    m = shape(W)[1]

    # make sure that W is symmetric, this is a computationally expensive operation, only use for debugging
    #if (W-W.transpose()).sum() != 0:
    #	print "W should be symmetric!"
    #	exit(0)

    # degrees and regularization
    # S Yu Understanding Popout through Repulsion CVPR 2001
    # Allows negative values as well as improves invertability
    # of d for small numbers
    # i bet that this is what improves the stability of the eigen
    d = abs(W).sum(0)
    dr = 0.5 * (d - W.sum(0))
    d = d + offset * 2
    dr = dr + offset

    # calculation of the normalized LaPlacian
    W = W + spdiags(dr, [0], m, m, "csc")
    Dinvsqrt = spdiags((1.0 / sqrt(d + eps)), [0], m, m, "csc")
    P = Dinvsqrt * (W * Dinvsqrt);
    return P


def to_rgb(x):
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb

def load_dataset(img_dir, annot_file, num_per_class=-1):
    data = []
    labels = []
    k = 0
    for filename in os.listdir(img_dir):
        if filename.endswith(".asm") or filename.endswith(".jpg") and k < 2: 
            label_filename = annot_file + filename.split('.')[0] + '.xml'
            tree = ET.parse(label_filename)
            root = tree.getroot()
            objects= tree.findall('.//object')
            for obj in objects:
                x = int(obj.find('.//xmin').text)
                y = int(obj.find('.//ymin').text)
                x_max = int(obj.find('.//xmax').text)
                y_max = int(obj.find('.//ymax').text)
                data.append(cv2.imread(img_dir + filename, cv2.COLOR_BGR2RGB)[y:y_max, x:x_max, :])
                # print(cv2.imread(img_dir + filename, cv2.COLOR_RGB2BGR)[y:y_max, x:x_max, :].shape)
            labels.append(1)
            k = k + 1
            continue
        else:
            continue
    return data, labels


# compute dense SIFT 
def computeSIFT(data):
    x = []
    for i in range(0, len(data)):
        sift = cv2.xfeatures2d.SIFT_create()
        img = data[i]
        step_size = 15
        kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
        dense_feat = sift.compute(img, kp)
        x.append(dense_feat[1])
        
    return x


def extract_denseSIFT(img):
    DSIFT_STEP_SIZE = 2
    sift = cv2.xfeatures2d.SIFT_create()
    disft_step_size = DSIFT_STEP_SIZE
    keypoints = [cv2.KeyPoint(x, y, disft_step_size)
            for y in range(0, img.shape[0], disft_step_size)
                for x in range(0, img.shape[1], disft_step_size)]

    descriptors = sift.compute(img, keypoints)[1]
    
    #keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


# form histogram with Spatial Pyramid Matching upto level L with codebook kmeans and k codewords
def getImageFeaturesSPM(L, img, kmeans, k):
    W = img.shape[1]
    H = img.shape[0]   
    h = []
    for l in range(L+1):
        w_step = math.floor(W/(2**l))
        h_step = math.floor(H/(2**l))
        x, y = 0, 0
        for i in range(1,2**l + 1):
            x = 0
            for j in range(1, 2**l + 1):                
                desc = extract_denseSIFT(img[y:y+h_step, x:x+w_step])                
                #print("type:",desc is None, "x:",x,"y:",y, "desc_size:",desc is None)
                predict = kmeans.predict(desc)
                histo = np.bincount(predict, minlength=k).reshape(1,-1).ravel()
                weight = 2**(l-L)
                h.append(weight*histo)
                x = x + w_step
            y = y + h_step
            
    hist = np.array(h).ravel()
    # normalize hist
    dev = np.std(hist)
    hist -= np.mean(hist)
    hist /= dev
    return hist


# get histogram representation for training/testing data
def getHistogramSPM(L, data, kmeans, k):    
    x = []
    for i in range(len(data)):        
        hist = getImageFeaturesSPM(L, data[i], kmeans, k)        
        x.append(hist)
    return np.array(x)


# build BoW presentation from SIFT of training images 
def clusterFeatures(all_train_desc, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_train_desc)
    return kmeans


def saliency_map_from_set(imgs):
    maps = []
    for image in imgs:
        # image=io.imread(ipFileName)
        # print("Image being read.")
        # image_uniqueness=np.zeros((image.shape[0],image.shape[1],3))
        # image_distribution=np.zeros((image.shape[0],image.shape[1],3))
        # image_saliency=np.zeros((image.shape[0],image.shape[1],3))
        # image=io.imread(image)
        # print(image.shape)

        colors=[]
        positions=[]

        # print("Started Abstraction.")
        colors,positions,seg,d_p,o,d_u = abstract(image)
        # print("Abstraction successful.")

        # print("Started Uniqueness Assignment.")
        Uniqueness=uniquenessAssignment(colors,positions)
        U_norm=Uniqueness/max(Uniqueness)
        # print("Uniqueness Assignment successful.")

        # print("Starting Distribution Assignment.")
        dist=distributionAssignment(colors,positions)
        dist_norm=dist/max(dist)
        # print("Distribution Assignment successful.")

        # print("Starting Saliency Assignment.")
        sal=saliency_Assignment(U_norm,dist_norm,colors,positions)    
        # print("Saliency Assignment successful.")

        # for i in range(len(d_p)):
        #     for k in range(len(d_p[i])):
                
        #         row=d_p[i][k][0]
        #         col=d_p[i][k][1]
        #         image_uniqueness[row,col]=Uniqueness[i]
        #         image_distribution[row,col]=dist_norm[i]
        #         image_saliency[row,col]=sal[i]
        maps.append(sal)
    return maps

def compute_purity(C_computed,C_grndtruth,R):
    """
    Clustering accuracy can be defined with the purity measure, defined here:
      Yang-Hao-Dikmen-Chen-Oja, Clustering by nonnegative matrix factorization
      using graph random walk, 2012.

    Usages:
      accuracy = compute_clustering_accuracy(C_computed,C_grndtruth,R)

    Notations:
      n = nb_data

    Input variables:
      C_computed = Computed clusters. Size = n x 1. Values in [0,1,...,R-1].
      C_grndtruth = Ground truth clusters. Size = n x 1. Values in [0,1,...,R-1].
      R = Number of clusters.

    Output variables:
      accuracy = Clustering accuracy of computed clusters.
    """

    N = C_grndtruth.size
    nb_of_dominant_points_in_class = np.zeros((R, 1))
    w = defaultdict(list)
    z = defaultdict(list)       
    for k in range(R):
        for i in range(N):
            if C_computed[i]==k:
                w[k].append(C_grndtruth[i])
        if len(w[k])>0:
            val,nb = stats.mode(w[k])
            z[k] = int(nb.squeeze()) 
        else:
            z[k] = 0
    sum_dominant = 0
    for k in range(R):
        sum_dominant = sum_dominant + z[k]
    purity = float(sum_dominant) / float(N)* 100.0
    return purity

def compute_ncut(W, Cgt, R):
    """
    Graph spectral clustering technique NCut:
      Yu-Shi, Multiclass spectral clustering, 2003
      Code available here: http://www.cis.upenn.edu/~jshi/software

    Usages:
      C,acc = compute_ncut(W,Cgt,R)

    Notations:
      n = nb_data

    Input variables:
      W = Adjacency matrix. Size = n x n.
      R = Number of clusters.
      Cgt = Ground truth clusters. Size = n x 1. Values in [0,1,...,R-1].

    Output variables:
      C = NCut solution. Size = n x 1. Values in [0,1,...,R-1].
      acc = Accuracy of NCut solution.
    """

    # Apply ncut
    eigen_val, eigen_vec = ncut( W, R )
    
    # Discretize to get cluster id
    eigenvec_discrete = discretisation( eigen_vec )
    res = eigenvec_discrete.dot(np.arange(1, R + 1)) 
    C = np.array(res-1,dtype=np.int64)
    
    # Compute accuracy
    computed_solution = C
    ground_truth = Cgt
    acc = compute_purity( computed_solution,ground_truth, R)

    return C, acc
