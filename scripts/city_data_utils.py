"""
:description: Utilities for loading in, preprocessing, and organizing data.
"""
import cv2
import glob
import numpy as np
import os
import random
import scipy.io

# filename and directory constants
DATA_DIR = '../data/cities'
OUTPUT_DIR = '../data/cities'

# data split constants
TRAIN_RATIO = .8
VAL_RATIO = .1
TEST_RATIO = .1

def get_city_directories(dataset_dir):
    pattern = os.path.join(dataset_dir, '*/')
    filepaths = glob.glob(pattern)
    return filepaths

def get_all_image_filepaths(dataset_dir, pattern):
    pattern = os.path.join(dataset_dir, pattern)
    filepaths = glob.glob(pattern)
    filenames = [filepath[filepath.index('cities/') + 7:] for filepath in filepaths]
    return filepaths, filenames
    
def generate_train_split(labels):
    # randomly determine the set of train locations
    train_split = set()
    for filepath, label in labels.iteritems():
        if random.random() < TRAIN_RATIO:
            train_split.add(filepath)
    return train_split

def get_city_name_from_city_directory(city_directory):
    city_name = city_directory.split('/')[-2]
    return city_name
    
def write_labels_file(dataset_dir, train_output_filepath, val_output_filepath, test_output_filepath, city_labels_filepath):

    # load the city directories e.g., '../data/cities/Barcelona/'
    city_directories = sorted(get_city_directories(dataset_dir))
    
    # populate a dictionary with the label of each image
    image_labels = dict()
    city_labels = dict()
    for city_idx, city_directory in enumerate(city_directories):

        city_name = get_city_name_from_city_directory(city_directory)
        city_labels[city_name] = city_idx
        filepaths, filenames = get_all_image_filepaths(city_directory, pattern='*.jpg')
        for filename in filenames:
            image_labels[filename] = city_idx

    # randomly generate the train split (and implictly the validation split)
    # not by image but by location
    train_split = generate_train_split(image_labels)

    # format the lines for each data entry
    train_lines = []
    val_lines = []
    filepaths, filenames = get_all_image_filepaths(dataset_dir, pattern='*/*.jpg')
    for filepath, filename in zip(filepaths, filenames):

        # if filename in the labels then write it
        if filename in image_labels:    
            label = image_labels[filename]
        else:
            print 'cant find label'
            continue
        
        string = '{} {}\n'.format(filename, label)
        if filename in train_split:
            train_lines.append(string)
        else:
            val_lines.append(string)        

    # write city labels to file
    with open(city_labels_filepath, 'wb') as f:
        for city, cidx in city_labels.iteritems():
            f.write('{} {}\n'.format(city, cidx))
            
    # write train lines to file
    with open(train_output_filepath, 'wb') as f:
        f.writelines(np.random.permutation(train_lines))
        
    # split val into test and val using mid point
    mid = len(val_lines) / 2

    # write val lines to file
    with open(val_output_filepath, 'wb') as f:
        f.writelines(np.random.permutation(val_lines[:mid]))

    # write test lines to file
    with open(test_output_filepath, 'wb') as f:
        f.writelines(np.random.permutation(val_lines[mid:]))

if __name__ == '__main__':
    train_dir = os.path.join(OUTPUT_DIR)
    train_data_file = os.path.join(OUTPUT_DIR, 'train.txt')
    val_data_file = os.path.join(OUTPUT_DIR, 'val.txt')
    test_data_file = os.path.join(OUTPUT_DIR, 'test.txt')
    city_labels_file = os.path.join(OUTPUT_DIR, 'city_labels.txt')
    assert os.path.isdir(train_dir), 'train directory: {} not found'.format(train_dir)
    if os.path.exists(train_data_file):
        open(train_data_file, 'a').close()
    if os.path.exists(val_data_file):
        open(val_data_file, 'a').close()
    if os.path.exists(test_data_file):
        open(test_data_file, 'a').close()
    if os.path.exists(city_labels_file):
        open(city_labels_file, 'a').close()

    write_labels_file(train_dir, train_data_file, val_data_file, test_data_file, city_labels_file)

    
