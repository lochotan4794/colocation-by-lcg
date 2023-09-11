
import numpy as np
from displayResult import displayResult


def select_box(imgs, M, z, pairs, boxes):
  zipped = list(zip(z, boxes))
  nb = len(imgs)
  for it in range(nb):
    zit = zipped[it * M : M*(it + 1)]
    sort_zipped = sorted(zit, key=lambda x: x[0])[-1]
    zi, bi = sort_zipped
    pairs[imgs[it]].append(bi)
  return pairs



def drawBox(dataset, Z, config):
    M = config['M']
    data_dir = config['data_folder']
    imgs = dataset.images
    box_coordinates = dataset.box_coordinates
    pairs = {}
    for i in range(len(imgs)):
        pairs[imgs[i]] = []
    select_box(imgs, M, Z, pairs, box_coordinates)
    displayResult(imgs, data_dir, pairs)
