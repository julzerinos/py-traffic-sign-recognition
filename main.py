import os
import random

import argparse

import cv2 as cv
import numpy as np

from imutils import rotate_bound

import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from scipy import ndimage
#import tensorflow as tf
#from tensorflow.compat.v1.layers import flatten
from sklearn.model_selection import train_test_split

from sign_mapping import german_mapping as mapping


VERBOSE = False
LIMIT = 0


def prepare_data(dataset='dataset/german'):
    signs = dict()

    signs_f = []
    signs_lb = []

    if VERBOSE:
        print("[Preproccessing] Loading all images")

    for sign in mapping.keys():
        sign_path = os.path.join(dataset, sign)
        signs[sign] = [
            os.path.join(sign_path, p) for p in sorted(os.listdir(sign_path))
            if '.ppm' in p
        ]

    if VERBOSE:
        print(
            "[Preproccessing] Loaded images, number of classifications:",
            len(signs)
        )
        for sign in signs:
            print(
                f"[Preproccessing] Label: {mapping[sign]} ({len(signs[sign])})"
                )

    for sign in signs:
        rpl = replicate(signs[sign], LIMIT)
        signs_f.extend(rpl)
        signs_lb.extend([sign] * len(rpl))
        if VERBOSE:
            print(
                "[Preproccessing] Set replicated, final length:",
                len(rpl),
                mapping[sign]
                )

    tmp_s, test_set, tmp_l, test_set_labels = train_test_split(
        signs_f, signs_lb,
        train_size=0.8, random_state=None
    )

    train_set, val_set, train_set_labels, val_set_labels = train_test_split(
        tmp_s, tmp_l,
        train_size=0.8, random_state=None
    )

    if VERBOSE:
        print(
            "\n"
            "[Preprocessing] Prepared image sets.",
            f"Test set ({len(test_set)}),",
            f"Validation set ({len(val_set)}),",
            f"Train set ({len(train_set)})"
        )

    return test_set, test_set_labels, \
        train_set, train_set_labels, \
        val_set, val_set_labels


def replicate(set, limit=0):
    images = [cv.imread(imgfn, 0) for imgfn in set]

    if len(images) >= limit:
        return images

    while len(images) < limit:
        images.append(
            apply_random_filter(random.choice(set))
        )

    return images


def apply_random_filter(imgfn):
    img = cv.imread(imgfn, 1)

    if VERBOSE:
        print("[Preproccessing]", imgfn, end='\r')

    # Random rotate
    img = rotate_bound(img, random.getrandbits(4) - 7)
    # Random brightness

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    random_bright = .25+np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    # # 50/50 blur or sharpen
    # if random.getrandbits(1) == 1:
    #     img = cv.medianBlur(img, 3)
    # else:
    #     kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #     img = cv.filter2D(img, -1, kernel)

    # Return as grayscale (monochannel)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def show_set(set):
    """Helper function which displays images from a set.
    Use , and . to switch images left and right.
    Use c to exit.
    """

    i = 0

    while True:
        cv.imshow('test', set[i])

        key = cv.waitKey(0)
        if key == ord('.') and i < len(set):
            i += 1
        elif key == ord(',') and i > 0:
            i -= 1
        elif key == ord('c'):
            break

def augment_brightness_camera_images(image):
    image1 = cv.cvtColor(image,cv.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv.cvtColor(image1,cv.COLOR_HSV2RGB)
    return image1

def transform_image(img):    
    ang_range = 25
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
        
    img = cv.warpAffine(img,Rot_M,(cols,rows))    
    img = augment_brightness_camera_images(img)
    
    return img
##cos sie zepsulo i nie slychac tutaj
def get_random_image_of_given_label(images_set, labels_set, label):
    numpy_indexes = np.array(labels_set)
    image_indexes = list(numpy_indexes).index(label)
    rand_index = random.randint(0, np.bincount(labels_set)[label] - 1)
    return images_set[image_indexes][rand_index]

def equalize_samples_set(X_set, y_set):
    labels_count_arr = np.bincount(y_set)
    labels_bins = np.arange(len(labels_count_arr))
    
    ind = 0    
   
    for label in tqdm(labels_bins):        
        labels_no_to_add =  int(np.mean(labels_count_arr)) * 4 - labels_count_arr[label]
        
        ind = ind + 1
        X_temp = []
        y_temp = []
        
        for num in range(labels_no_to_add):      
            rand_image = get_random_image_of_given_label(X_set, y_set, label)
            X_temp.append(transform_image(rand_image))
            y_temp.append(label)            
   
        X_set = np.append(X_set, np.array(X_temp), axis=0)
        y_set = np.append(y_set, np.array(y_temp), axis=0)
        
    return X_set, y_set

if __name__ == '__main__':

    # Control line arguments
    # '--verbose' to display all steps
    # '--replimit' sets limit on image replication
    parser = argparse.ArgumentParser(
            description='Traffic sign detection algorithm.'
        )
    parser.add_argument(
        '--verbose', action='store_true',
        help='increase verbosity (display all actions)'
        )
    parser.add_argument(
        '--replimit', '-l', action='store', default=3000, type=int, dest='lim',
        help="Define maximum set size after replication (0 for no replication)"
        )
    args = parser.parse_args()
    VERBOSE = args.verbose
    LIMIT = args.lim

    # Step 1: Load data
    # Data is loaded into sets
    # each _set is a random collection of images
    # each _set_label is that image's respective label
    test_set, test_set_labels, \
        train_set, train_set_labels, \
        valid_set, valid_set_labels = prepare_data()

    #show_set(test_set)

    train_set, train_set_labels = equalize_samples_set(train_set, train_set_labels)
    
 

    # Step 2: Dataset Summary & Exploration (not necessary?)
    # number of training/validation/test examples etc
    # images and histograms

    # Step 3: Design and Test a Model Architecture
    # Equalize histograms of training samples - by generation of additional, transformed images
    # Image normalization and grayscale conversion
    # Model Architecture
    # Train, Validate and Test the Model
    # Train accuracy

    # Step 4: Test a Model on New Images
    # Load and Output the Images
    # Predict the Sign Type for Each Image
    # Analyze Performance
