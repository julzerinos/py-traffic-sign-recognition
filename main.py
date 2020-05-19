import os

import cv
from sklearn.model_selection import train_test_split

from sign_mapping import mapping

def prepare_data(dataset='dataset/german'):
    signs_fn = []
    signs_lb = []

    for sign in mapping.keys():
        sign_path = os.path.join(dataset, sign)
        for p in sorted(os.listdir(sign_path)):
            signs_fn.append(os.path.join(sign_path, p))
            signs_lb.append(sign)

    tmp_s, test_set, tmp_l, test_set_labels = train_test_split(
        signs_fn, signs_lb,
        train_size=0.8, random_state=None
    )

    train_set, valid_set, train_set_labels, valid_set_labels = train_test_split(
        tmp_s, tmp_l,
        train_size=0.8, random_state=None
    )

    return test_set, test_set_labels, train_set, train_set_labels, valid_set, valid_set_labels

def parse():
    pass


if __name__ == '__main__':

    # Step 1: Load data
    # Data is loaded into sets
    # each _set is a random collection of filenames of images
    # each _set_label is that image's respective label

    test_set, test_set_labels, train_set, train_set_labels, valid_set, valid_set_labels = prepare_data()

    print(test_set[0])


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
