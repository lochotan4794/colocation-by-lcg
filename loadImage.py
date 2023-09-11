import numpy as np
import cv2
from runObjectness import run_objectness
import os
from defaultParams import default_params

class ImageDataset:

    def __init__(self, config_dictionary):
        self.config_dictionary = config_dictionary
        self.image_root = config_dictionary['data_folder']
        self.len_dataset = config_dictionary['Number_of_images']
        self.images = None
        self.box_data = None
        self.box_coordinates = None
        self.load_from_folder(self.image_root)
  
    # will move to utils later
    def load_from_folder(self, root):
        self.images = os.listdir(root)
        print(self.images)

    def objectness(self):
        box_data = []
        params = default_params('.')
        params.cues = ['SS']
        box_coordinates = []
        for img in self.images:
            img_id =  self.image_root + img
            print(img_id)
            img_example = cv2.imread(img_id)[:, :, ::-1]
            boxes = run_objectness(img_example, self.config_dictionary['M'], params)
            print('objectness done')
            box_coordinates = box_coordinates + boxes.tolist()
            box_data += self.boxes_data_from_img(boxes, img_id)
        self.box_data = box_data
        self.box_coordinates = box_coordinates

    def round_box(self, box):
        xmin, ymin, xmax, ymax, score = box
        return (int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax)), score)

    def boxes_data_from_img(self, boxes, img_id):
        box_data = []
        img_data = cv2.imread(img_id, cv2.COLOR_BGR2RGB)
        for box in boxes:
            xmin, ymin, xmax, ymax, score = self.round_box(box)
            box_data.append(img_data[ymin:ymax, xmin:xmax, :])
        return box_data
    