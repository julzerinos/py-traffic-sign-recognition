import os
import logging, sys
import random

import argparse

import cv2 as cv
import numpy as np

from imutils import rotate

from sklearn.model_selection import train_test_split

from sign_mapping import german_mapping as mapping


VERBOSE = False


def prepare_data(dataset='dataset/german'):
    signs = dict()

    signs_f = []
    signs_lb = []

    if VERBOSE: print("[Preproccessing] Loading all images")

    for sign in mapping.keys():
        sign_path = os.path.join(dataset, sign)
        signs[sign] = [
            os.path.join(sign_path, p) for p in sorted(os.listdir(sign_path))
            if '.ppm' in p
        ]

    if VERBOSE:
        print("[Preproccessing] Loaded images, number of classifications:", len(signs))
        for sign in signs: print(f"[Preproccessing] Label: {mapping[sign]} ({len(signs[sign])})")
        print()

    for sign in signs:
        rpl = replicate(signs[sign])
        signs_f.extend(rpl)
        signs_lb.extend([sign] * len(rpl))
        if VERBOSE: print(f"[Preproccessing] Set replicated, final length:", len(rpl), mapping[sign])

    tmp_s, test_set, tmp_l, test_set_labels = train_test_split(
        signs_f, signs_lb,
        train_size=0.8, random_state=None
    )

    train_set, valid_set, train_set_labels, valid_set_labels = train_test_split(
        tmp_s, tmp_l,
        train_size=0.8, random_state=None
    )

    if VERBOSE: print(
        "\n"
        f"[Preprocessing] Prepared image sets.",
        f"Test set ({len(test_set)}),",
        f"Validation set ({len(valid_set)}),",
        f"Train set ({len(train_set)})"
        )

    return test_set, test_set_labels, \
        train_set, train_set_labels, \
        valid_set, valid_set_labels

def replicate(set, limit = 3000):
    images = [cv.imread(imgfn) for imgfn in set]

    if len(images) >= limit:
        return images

    while len(images) < limit:
        images.append(
            apply_random_filter(random.choice(set))
        )


    return images

def apply_random_filter(imgfn):
    img = cv.imread(imgfn)

    if VERBOSE: print("[Preproccessing]", imgfn, end='\r')

    # Random rotate
    img = rotate(img, random.getrandbits(4) - 7)
    # Random brightness
    img = cv.add(img, np.array([float(random.getrandbits(7) - 64)]))

    # 50/50 blur or sharpen
    if random.getrandbits(1) == 1:
        img = cv.medianBlur(img, 3)
    else:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv.filter2D(img, -1, kernel)

    # Return as grayscale (monochannel)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


if __name__ == '__main__':

    # Control line arguments
    # Setup '--verbose' to display all steps
    parser = argparse.ArgumentParser(
            description='Traffic sign detection algorithm.'
        )
    parser.add_argument('--verbose', action='store_true',
                       help='increase verbosity (display all actions)')
    VERBOSE = parser.parse_args().verbose

    # Step 1: Load data
    # Data is loaded into sets
    # each _set is a random collection of images
    # each _set_label is that image's respective label
    test_set, test_set_labels, \
        train_set, train_set_labels, \
        valid_set, valid_set_labels = prepare_data()


##Step 2: Dataset Summary & Exploration (not necessary?)

#number of training/validation/test examples etc

#images and histograms

##Step 3: Design and Test a Model Architecture

#Equalize histograms of training samples - by generation of additional, transformed images

#Image normalization and grayscale conversion

#Model Architecture

#Train, Validate and Test the Model

#Train accuracy

##Step 4: Test a Model on New Images

#Load and Output the Images

#Predict the Sign Type for Each Image

#Analyze Performance
