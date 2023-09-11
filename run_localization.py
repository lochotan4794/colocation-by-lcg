from runObjectness import run_objectness
# import cv2
from defaultParams import default_params
from drawBoxes import draw_boxes
from computeObjectnessHeatMap import compute_objectness_heat_map
import time
from utils import *
# load training dataset
# from sklearn.cluster import KMeans
# from sklearn import preprocessing
from LCG_optimizer import *
from spm import * 
# from scipy.optimize import check_grad, approx_fprime
import numdifftools as nd
from displayResult import displayResult
import math
from extract_features import load_train_all, extract_feature
from sklearn.metrics.pairwise import cosine_similarity
from LPCG_optimizer import LPCG_Optimizer 
np.random.seed(34)

# imgs = ['000001.jpg', '000003.jpg', '000002.jpg']
# data_dir = '/Users/admin/Documents/Colocalization/PC07/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
imgs = ['cat1.jpg', 'cat2.jpg', 'cat3.jpg']
data_dir = '/Users/admin/Documents/Colocalization/examples/'
params = default_params('.')
mu = 0.6
lamda = 0.001
nb = 10
M = 5

# with open('A.npy', 'rb') as a:
#     A = np.load(a)

# with open('L.npy', 'rb') as l:
#     L = np.load(l)

# with open('prior.npy', 'rb') as p:
#     box_prior = np.load(p)
#     if box_prior.ndim == 1:
#        box_prior = np.expand_dims(box_prior, 1)

# with open('coord.npy', 'rb') as p:
#     box_coordinates = np.load(p)

params.cues = ['SS']
# Feature Representation
# extract dense sift features from training images

# M: number of candidate boxes
M = 5
tic = time.time()
print("Objectness ....")
boxes_data, box_coordinates = extract_boxes(run_objectness, M, params, data_dir, imgs)
# toc = time.time()

print("Load trainning data ....")
train_data = load_train_all(boxes_data)
print("Feature extraction ....")
X = extract_feature(train_data)

def _box_similarity_matrix(X, M):
    (nb, d) = X.shape
    s_matrix = np.zeros(shape=(nb, nb))   
    i_array = [int(id / M) for id in range(nb)]
    gama = 1 / np.sqrt(10*d)
    for i in range(nb):
            for j in range(nb):
                if i_array[i] != i_array[j]:
                    x = np.sum(np.power((X[i, :] - X[j, :]), 2) / (X[i, :] + X[j, :]))
                    s_matrix[i, j] = np.exp(-gama * x)
    return s_matrix

def _sparse_sim_matrix(S, M, nb):
    copied_matrix = np.copy(S)
    assign_box = [int(id / M) for id in range(nb)]
    for i in range(nb):
        for j in range(nb):
            if assign_box[i] == assign_box[j]:
                copied_matrix[i, j] = 0
    return copied_matrix

# nb: total number of boxes
nb = len(boxes_data)
print("X shape {}".format(X.shape))
spm =  SpatialPyramidClassifier()
X = spm.get_descriptors(train_data, [1 for i in range(nb)])

print("Similarity matrix ...")
S = cosine_similarity(X)
similarity_matrix = _sparse_sim_matrix(S, M, nb)
# similarity_matrix = _box_similarity_matrix(X, M)
print("Similarity matrix {}".format(similarity_matrix))

def box_prior(boxes_data):
    saliency = saliency_map_from_set(boxes_data)
    # print(saliency)
    v = []
    for s in saliency:
        # In paper, it is weighted by score, will fix later
        v.append(np.mean(s))
    return np.array(v)
# https://github.com/jiviteshjain/bag-of-visual-words/blob/main/src/bovw.ipynb
# Co-localization in Real-World Images

print("Box prior ...")
tic = time.time()
box_prior = box_prior(boxes_data)
if box_prior.ndim == 1:
       box_prior = np.expand_dims(box_prior, 1)
toc = time.time()

print("Box prior time: {}".format(toc - tic))
print("Compute Box similarity ...")
L = normalize_laplacian(similarity_matrix)
print("Compute Box similarity shape {}".format(L))
print("L shape {}".format(L.shape))
print("prior m shape {}".format(box_prior.shape))

# number of class
k = 2
# central projection matrix
# Vector ones
vector_one = np.ones(shape=(nb,1))
n_class = 2
# Indicator matrix
y = np.array([0, 1])

# https://rich-d-wilkinson.github.io/MATH3030/2.4-centering-matrix.html
# I = (labels[:,None]==y).astype(int)

dim = X.shape[1]
I = np.identity(nb)
print("Indicator matrix {}".format(I.shape))
print("Centering projection matrix ...")
central_matrix = I - (1/nb) * np.dot(vector_one, vector_one.T)
# nbox is number of box

# # define function evaluation oracle
# X = cls() 
b = np.ones(shape=(nb, nb))
print("Box disimilarity")
Identity = np.identity(dim)
# Box dismimilarity
A = discriminative_optimial(central_matrix, X, nb, Identity, k)
# # z: binary variable (n , m)
# # n: number of box
# # m: number of candidate box

# with open('L.npy', 'wb') as l:
#     np.save(l, L)
# l.close()

# with open('A.npy', 'wb') as a:
#     np.save(a, A)
# a.close()

# with open('prior.npy', 'wb') as p:
#     np.save(p, box_prior)
# p.close()

# with open('coord.npy', 'wb') as c:
#     np.save(c, box_coordinates)
# c.close()

nb = 10

tmp = L + mu*A

print("tmp shape {}".format(tmp.shape))

print("L shape {}".format(L.shape))

print("A shape {}".format(A.shape))

print("prior shape {}".format(box_prior.shape))

def f(x):
    #t1: mdim, nbox
    # x = x.reshape((nbox, 1))
    t1 = np.dot(x.T, tmp)
    #t2: mxim, mdim
    # print("t1.shapef= {}".format(t1.shapt3)
    t2 = np.dot(t1, x)
    # print("t2.shape = {}".format(t2.shape))
    term = np.dot(x.T, np.log(box_prior))
    term = np.array(np.dot(x.T, np.log(box_prior)))
    t3 = np.array(t2 - np.array(lamda * term))
    return t3[0,0]

def grad_f(x):
    # grad = nd.Gradient(f)(x)
    term = np.log(box_prior)
    grad = np.dot(tmp, x) - lamda * term
    return grad

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
#X0 = np.ones(shape=(nb, 1))
X0 = np.random.choice([0, 1], size=(nb, 1), p=[1./2, 1./2])
#X0 = np.random.rand((nb, 1))

#solution, logs = LCG_optimizer(X0, 1000, f, grad_f)
solution, logs = LPCG_Optimizer(X0, f, grad_f, 1000)


plt.plot([i for i in range(len(logs))], logs)
plt.savefig("logs.jpg")

print("Solution: {}".format(solution))

pairs = {}

def select_box(images, M, z, pairs, boxes):
  zipped = list(zip(z, boxes))
  nb = len(images)
  for it in range(nb):
    zit = zipped[it * M : M*(it + 1)]
    # print(zit)
    sort_zipped = sorted(zit, key=lambda x: x[0])[-1]
    zi, bi = sort_zipped
    pairs[imgs[it]].append(bi)
  return pairs

for i in range(len(imgs)):
    pairs[imgs[i]] = []

# for p in range(len(solution)):
#     id = math.floor(p / M)
#     pairs[imgs[id]].append(box_coordinates[p])

select_box(imgs, M, solution, pairs, box_coordinates)

print(pairs)
displayResult(imgs, data_dir, pairs)
