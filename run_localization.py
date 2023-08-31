from runObjectness import run_objectness
import cv2
from defaultParams import default_params
from drawBoxes import draw_boxes
from computeObjectnessHeatMap import compute_objectness_heat_map
import time
from utils import *
# load training dataset
from sklearn.cluster import KMeans
from sklearn import preprocessing
from LCG_optimizer import *
from spm import * 
from scipy.optimize import check_grad, approx_fprime
import numdifftools as nd
from displayResult import displayResult
import math

# imgs = ['000003.jpg', '000002.jpg']
# data_dir = '/Users/admin/Documents/Colocalization/PC07/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
imgs = ['cat1.jpg', 'cat2.jpg']
data_dir = '/Users/admin/Documents/Colocalization/examples/'

params = default_params('.')
# params.cues = ['SS']

# tic = time.time()
# boxes = run_objectness(img_example, 10, params)
# toc = time.time()

# print("%f" % (toc - tic))
# draw_boxes(img_example, boxes, base_color=(1, 0, 0))
# compute_objectness_heat_map(img_example, boxes)

# Feature Representation
# extract dense sift features from training images

# M: number of candidate boxes
M = 5
tic = time.time()
# boxes = run_objectness(img_example, 10, params)
train_data, box_coordinates = extract_boxes(run_objectness, M, params, data_dir, imgs)
toc = time.time()

print('Generate box ...')
x_train = computeSIFT(train_data)
print("SIFT feature {}".format(len(x_train)))
# x_test = computeSIFT(test_data)
scale_parameter = 1
# Commbine 
all_train_desc = []
for i in range(len(x_train)):
    for j in range(x_train[i].shape[0]):
        all_train_desc.append(x_train[i][j,:])

all_train_desc = np.array(all_train_desc)
# Quantilized
# distinctive image features from scale-invariant keypoints: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
print("Feature Representation")
#number of clusters
k = 10
# Window size
L = 2
kmeans = clusterFeatures(all_train_desc, k)
train_histo = getHistogramSPM(L, train_data, kmeans, k)
# test_histo = getHistogramSPM(2, test_data, kmeans, k)
# Pool the sift feature
# Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories  https://inc.ucsd.edu/mplab/users/marni/Igert/Lazebnik_06.pdf
print("getting feature ...")
X = train_histo
print("X shape {}".format(X.shape))
# spm =  SpatialPyramidClassifier()
# X = spm.get_descriptors(train_data, train_label)
print("Similarity matrix ...")
similarity_matrix = np.corrcoef(X)
print("Similarity matrix shape {}".format(similarity_matrix.shape))
def box_prior(train_data):
    saliency = saliency_map_from_set(train_data)
    v = []
    for s in saliency:
        # In paper, it is weighted by score, will fix later
        v.append(np.mean(s))
    return np.array(v)

# Co-localization in Real-World Images
print("Box prior ...")
box_prior = box_prior(train_data)

# print("Silient map shape... {}".format(box_prior.shape))
for b in box_prior:
    print(b.shape)

print("Compute Box similarity ...")
L = normalize_laplacian(similarity_matrix, 1)
print("Compute Box similarity shape {}".format(similarity_matrix.shape))
# nb: total number of boxes
nb = len(x_train)
# number of class
k = 2
# central projection matrix
# Vector ones
vector_one = np.ones(shape=(nb,1))
n_class = 2
# Indicator matrix
y = np.array([0, 1])
labels = np.array([1 for i in range(nb)])
print("labels: {}".format(labels))
# https://rich-d-wilkinson.github.io/MATH3030/2.4-centering-matrix.html
# I = (labels[:,None]==y).astype(int)
dim = X.shape[1]
I = np.identity(nb)
print("Indicator matrix {}".format(I.shape))
print("Centering projection matrix ...")
central_matrix = I - (1/nb) * np.dot(vector_one, vector_one.T)
# nbox is number of box
mu = 0.5
lamda = 0.1
# # define function evaluation oracle
# X = cls() 
b = np.ones(shape=(nb, nb))
print("Box disimilarity")
Identity = np.identity(dim)
# Box dismimilarity
A = discriminative_optimial(central_matrix, X, nb, Identity, k)
# z: binary variable (n , m)
# n: number of box
# m: number of candidate box
tmp = L + mu*A
# def f(z):
#     print("z shape {}".format(z.shape))
#     print("L shape {}".format(L.shape))
#     print("A shape {}".format(A.shape))
#     # tmp: nbox, nbox
#     # z: nbox, dim
#     # print(box_prior)
#     return np.sum(np.dot(np.dot(z.T, tmp.reshape(-1)), z) - lamda * np.dot(z.T, np.log(box_prior)))
# print("z shape {}".format(x.shape))
print("L shape {}".format(L.shape))
print("prior m shape {}".format(box_prior.shape))

def f(x):
    #t1: mdim, nbox
    # x = x.reshape((nbox, 1))
    t1 = np.dot(x.T, tmp)
    #t2: mxim, mdim
    # print("t1.shapef= {}".format(t1.shapt3)
    t2 = np.dot(t1, x)
    # print("t2.shape = {}".format(t2.shape))
    t3 = t2 - lamda * np.dot(x.T, np.log(box_prior))
    return t3[0,0]

def grad_f(x):
    # grad = nd.Gradient(f)(x)
    grad = np.dot(x.T, A) - lamda * np.log(box_prior)
    return grad.T

# define a model class as feasible region
class Model_l1_ball:

    def __init__(self, size):
      self.size = size

    def minimize(self, gradient_at, x):
      result = np.zeros(self.size)
      if gradient_at(x) is None:
          result[0] = 1
      else:
          i = np.argmax(np.abs(gradient_at(x)))
          result[i] = -1 if gradient_at(x)[i] > 0 else 1
      return result
    
# def grad_f(z):
#     # grad = nd.Gradient(f)([x.reshape(-1)])
#     grad = np.sum(np.dot(z.T, tmp) - lamda * np.log(box_prior))
#     print("Gradient shape {}".format(grad))
#     return grad

# def grad_2_f(x):
#     print(np.sum(tmp - lamda).shape)
#     return np.sum(tmp - lamda)

def grad_2_f(x):
    print("caculate grad 2..")
    grad = nd.Gradient(grad_f)([x])
    # grad = A
    return grad


l1Ball = Model_l1_ball(A.shape[1])  # initialize the feasible region as a L1 ball
X0 = np.ones(shape=(nb, 1))

solution, logs = LCG_optimizer(X0, 10, f, grad_f)
# # print(logs)
# pairs = {'000001.jpg': [[0.0, 0.0, 345.64583333333337, 302.0833333333333, 0.9999999998469948]], '000002.jpg': []}
# print("solution: {}".format(solution))
# plt.plot([i for i in range(len(logs))], logs)
# plt.show()
# solution = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# M = 5
# train_data = [(12, 24, 23, 56) for i in range(10)]
pairs = {}
for i in range(len(imgs)):
    pairs[imgs[i]] = []

for p in range(len(solution)):
    if solution[p] > 0.8:
        id = math.floor(p / M)
        # print(p)
        # print(id)
        pairs[imgs[id]].append(box_coordinates[p])

print(pairs)
displayResult(imgs, data_dir, pairs)
