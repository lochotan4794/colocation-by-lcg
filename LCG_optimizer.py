# conditional autograd import
import importlib.util
spec = importlib.util.find_spec("autograd")
# if spec is None:
#     import numpy as np
# else:
#     import autograd.numpy as np
#     from autograd import grad

import numpy as np
import logging

# from .objective_functions.obj_function import ObjectiveFunction
# from .algorithm import Algorithm
# from . import utils
# from . import globs


# #TODO: Logging shouldn't be configured when run as a module, it is the
# #task of the main application.
# logging.basicConfig(level=logging.INFO,
#                     format='%(message)s')


# def init_model(model):
#     if isinstance(model, str):  # LP file
#         from .LPsolver import initModel_fromFile
#         feasible_region = initModel_fromFile(model)
#     else:  # user-defined model class
#         feasible_region = model
#     return feasible_region

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
  for Xs in cache:
    if np.dot(linear_obj.T, X - Xs) > acc:
      return Xs
  y = oracle(-linear_obj, size)
  if np.dot(linear_obj.T, X - y) > acc:
    cache.append(y)
    return y
  else:
    return False
  
  

# def LCG_optimizer(initial, max_iter, grad_f, grad_2_f):
#   Xs = [initial]
#   f_cache = []
#   y_cache = []
#   K = 1.1
#   phi = 5
#   for i in range(1, max_iter):
#     cc = grad_f(Xs[-1])
#     C = np.linalg.norm(grad_2_f(Xs[-1]) , ord=2) / np.linalg.norm(1 + cc**2, ord=2)**3
#     step = phi / K * C
#     phi = (phi + C * step**2) / (1 + step / K)
#     acc = phi / K
#     v = LPSep(cc, 8, Xs[-1], y_cache, acc)
#     if not isinstance(v, bool):
#       Xs.append(Xs[-1]  + step * (v - Xs[-1]))
#     else:
#       Xs.append(Xs[-1])
#     f_cache.append(f(Xs[-1]))
#   return Xs[-1], f_cache

def LCG_optimizer(initial, max_iter, f, grad_f):
  Xs = [initial]
  f_cache = []
  y_cache = []
  K = 1.1
  phi = 0
  for i in range(1, max_iter):
    step = 2 / (i + 2)
    cc = grad_f(Xs[-1])
    phi = (phi + 0.5 * step**2) / (1 + step / K)
    acc = phi / K
    v = LPSep(cc, cc.shape[0], Xs[-1], y_cache, acc)
    if not isinstance(v, bool):
      print("new step ...")
      Xs.append(Xs[-1]  + step * (v - Xs[-1]))
    else:
      Xs.append(Xs[-1])
    f_cache.append(f(Xs[-1]))
  return Xs[-1], f_cache



