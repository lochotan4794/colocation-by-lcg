import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Get image as numpy array
def load_image(name, path):
    img_path = path + name
    img = cv2.imread(img_path)
    return img

# Plot numpy array
def plot_image(img):
    plt.imshow(img)
    plt.title(img.shape)
    
# Plot a grid of examples
def plot_grid(img_names, img_root, rows=5, cols=5):
    fig = plt.figure(figsize=(25,25))
    
    for i,name in enumerate(img_names):
        fig.add_subplot(rows,cols,i+1)
        img = load_image(name, img_root)
        plot_image(img)
        
    plt.show()

def class_to_color(class_id):
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,100,100),
              (100,255,100),(100,100,255),(255,100,0),(255,0,100),(100,0,255),(100,100,255),(100,255,0),
              (100,255,100)]
    return colors[class_id]

# draw a single bounding box onto a numpy array image
def draw_bounding_box(img, annotation):
    if len(annotation) == 0:
        return
    print(annotation)
    x_min, y_min, x_max, y_max = int(annotation[0]), int(annotation[1]), int(annotation[2]), int(annotation[3])
    
    class_id = int(0)
    color = class_to_color(class_id)
    
    cv2.rectangle(img,(x_min,y_min),(x_max,y_max), color, 2)

# draw all annotation bounding boxes on an image
def annotate_image(img, name, all_annotations):
    annotations = all_annotations[name]
    for anno in annotations:
        draw_bounding_box(img, anno)

# Plot a grid of xray images, each with all its bounding boxes
def plot_annotated_grid(img_names, img_root, all_annotations, rows=5, cols=5):
    fig = plt.figure(figsize=(16,16))
    
    for i,name in enumerate(img_names):
        fig.add_subplot(rows,cols,i+1)
        img = load_image(name, img_root)
        annotate_image(img, name, all_annotations)
        plot_image(img)
        
    plt.show()

def displayResult(filenames, IMG_ROOT, boxes):
    plot_annotated_grid(filenames, IMG_ROOT, boxes, rows=5, cols=5)


