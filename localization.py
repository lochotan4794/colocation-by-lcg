 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-
import numpy as np
from get_features import Feature
from loadImage import ImageDataset
from draw_box import drawBox
from obj_function import ObjectiveFunction
from optimizer import Optimizer

# you can construct your own configuration dictionary
config_dictionary = {
        'data_folder': '/Users/admin/Documents/Colocalization/examples/',
        'cache_folder': '/Users/admin/Documents/Colocalization/cache/',
        'algo': 'LCPG',
        'Number_of_images': 2,
        'max_iter': 100,
        'M': 5,
        'C': 1,
        'K': 1,
        'use_cache': False,
        'mu': 0.1,
        'lamda': 0.006,
        'params': ['SS']
        }

dataset = ImageDataset(config_dictionary)

dataset.objectness()

FeatureConstructor = Feature(dataset, config_dictionary)

L, A, prior = FeatureConstructor.calc()

obj = ObjectiveFunction(L, A, prior, config_dictionary)

optimizer = Optimizer(obj, config_dictionary)

Z, Logs = optimizer.RUN()

drawBox(dataset, Z, config_dictionary)



