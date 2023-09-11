import numpy as np



# Implement Unix simplex polytope
def Unit_simplex(linear_obj, size, x):
  result = np.zeros(shape=(size, 1))
  nzIdx = np.nonzero(x)
  if linear_obj is None:
    result[0, 0] = 1
  else:
    i = np.argmax(linear_obj[nzIdx])
    result[i, 0] = 1
  return result


def oracle(linear_obj, size):
  result = np.zeros(size) 
  result = np.expand_dims(result, axis=1)
  if linear_obj is None:
    result[0] = 1
  else:
    i = np.argmax(np.abs(linear_obj))
    result[i] = -1 if linear_obj[i, 0] > 0 else 1
  return result

def LPSep(linear_obj, size, X, cache, acc):
  print("linear obj {}".format(linear_obj))
  for Xs in cache:
    # print("LO {}".format(np.dot(linear_obj.T, X - Xs)))
    print("X shape {}".format(X.shape))
    print("Xs shape {}".format(Xs.shape))
    if np.dot(linear_obj.T, X - Xs) > acc:
      print("old vertice")
      return Xs
  # y = oracle(-linear_obj, size)
  y = Unit_simplex(-linear_obj, size, X)

  print(y.shape)
  if np.dot(linear_obj.T, X - y) > acc:
    cache.append(y)
    print("new vertice")
    return y
  else:
    return False
  
def grad_i(x, grad_f):
    grad = []
    grad_x = grad_f(x)
    for i in range(x.shape[0]):
        if x[i, 0] > 0:
            grad.append(grad_x[i, 0])
        elif x[i, 0] == 0:
            grad.append(-100)
    result = np.array(grad)
    result = np.expand_dims(result, 1)
    return result


def LPCG_Optimizer(initial, f, grad_f,  maxiter = 10):
    dim = initial.shape[0]
    print("dim {}".format(dim))
    # intial step size
    step = 1
    # Upper bound 
    phi = 1
    Xs = [initial]
    f_cache = []
    # Caching solution
    y_cache = []
    # Curvature
    C = 1
    delta = 1
    K = 1.5
    f_cache.append(f(Xs[-1]))
    for t in range(maxiter):
        phi = (2*phi + step **2 * C) / (2 + step / (K * delta))
        # increasing step size
        step =  t + 2 
        #  LO on casterian product of P (PxP)
        away_grad = np.array(- grad_i(Xs[-1], grad_f))
        grad = np.array(grad_f(Xs[-1]))
        c = np.concatenate((grad, away_grad), axis=None)
        c = np.expand_dims(c, axis=1)
        cp = np.concatenate((Xs[-1], Xs[-1]), axis=None)
        cp = np.expand_dims(cp, axis=1)
        # vertice on PxP
        v = LPSep(c, 2*dim, cp, y_cache, phi / delta)
        if isinstance(v, bool):
            Xs.append(Xs[-1])
        else:        
            v1 = v[0:dim, 0]
            v1 = np.expand_dims(v1, axis=1)
            v2 = v[dim:dim*2, 0]
            v2 = np.expand_dims(v2, axis=1)
            print("v1 shape {}".format(v1.shape))
            print("v2 shape {}".format(v2.shape))
            mus = [1 / np.power(2, eps) for eps in range(int(step)) ]
            mu = max(mus)
            print("Xs[-1] shape {}".format(Xs[-1].shape))
            X_new = Xs[-1] + mu * (v1 - v2)
            # caching
            Xs.append(X_new)
        f_cache.append(f(Xs[-1]))
    return Xs[-1], f_cache