import numpy as np
from extract_features import *
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
from spm import *

class Feature:

    def __init__(self, dataset, config_dictionary):
        self.dataset = dataset
        self.config_dictionary = config_dictionary
        
    def _load_params(self, glob_dir):
        
        with open(glob_dir + 'A.npy', 'rb') as a:
            self.A = np.load(a)
        a.close()

        with open(glob_dir + 'L.npy', 'rb') as l:
            self.L = np.load(l)
        l.close()

        with open(glob_dir + 'prior.npy', 'rb') as p:
            self.box_prior = np.load(p)
        p.close()

        with open(glob_dir + 'coord.npy', 'rb') as c:
            self.box_coordinates = np.load(c)
        c.close()

    def _save_params(self, glob_dir):
        with open(glob_dir + 'L.npy', 'wb') as l:
            np.save(l, self.L)
        l.close()

        with open(glob_dir + 'A.npy', 'wb') as a:
            np.save(a, self.A)
        a.close()

        with open(glob_dir + 'prior.npy', 'wb') as p:
            np.save(p, self.box_prior)
        p.close()

        with open(glob_dir + 'coord.npy', 'wb') as c:
            np.save(c, self.box_coordinates)
        c.close()

    def _sparse_sim_matrix(self, S, M, nb):
        copied_matrix = np.copy(S)
        assign_box = [int(id / M) for id in range(nb)]
        for i in range(nb):
            for j in range(nb):
                if assign_box[i] == assign_box[j]:
                    copied_matrix[i, j] = 0
        return copied_matrix
    
    def box_prior(self, boxes_data):
        saliency = saliency_map_from_set(boxes_data)
        v = []
        for s in saliency:
        # In paper, it is weighted by score, will fix later
            v.append(np.mean(s))
            return np.array(v)
  
    def calc(self):
        box_data = self.dataset.box_data
        box_coordinates = self.dataset.box_coordinates
        # check error 
        bins = []
        self.A = 0
        self.L = 0
        self.box_prior = 0
        self.box_coordinates = 0
        M = self.config_dictionary['M']
        nb = len(box_data)

        if self.config_dictionary['use_cache']:
            self._load_params(self.config_dictionary['cache_folder'])
        else:
            train_data = load_train_all(box_data)
            print("Feature extraction ....")
            X = extract_feature(train_data)
            spm =  SpatialPyramidClassifier()
            X = spm.get_descriptors(train_data, [1 for i in range(nb)])
            S = cosine_similarity(X)
            similarity_matrix = self._sparse_sim_matrix(S, M, nb)

            # https://github.com/jiviteshjain/bag-of-visual-words/blob/main/src/bovw.ipynb
            # Co-localization in Real-World Images

            self.box_prior = box_prior(box_data)
            if box_prior.ndim == 1:
                box_prior = np.expand_dims(box_prior, 1)

            self.L = normalize_laplacian(similarity_matrix)
            # number of class
            k = 2
            # central projection matrix
            vector_one = np.ones(shape=(nb,1))

            # https://rich-d-wilkinson.github.io/MATH3030/2.4-centering-matrix.html
            # I = (labels[:,None]==y).astype(int)

            dim = X.shape[1]
            I = np.identity(nb)
            central_matrix = I - (1/nb) * np.dot(vector_one, vector_one.T)
            Identity = np.identity(dim)
            # Box dismimilarity
            self.A = discriminative_optimial(central_matrix, X, nb, Identity, k)
            self._save_params(self.config_dictionary['cache_folder'])

        return self.L, self.A, self.box_prior



        

