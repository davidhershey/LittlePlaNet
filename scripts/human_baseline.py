

import cv2
import numpy as np
import os
import random
import shutil
import urllib

import data_utils

LABELS_URL = 'http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/GPS_Long_Lat_Compass.mat'
URL = 'http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/images/{}_{}.jpg'
DATASET_DIR = 'images'
MIN_ID = 1
MAX_ID = 10000
PERSPECTIVES = [0,1,2,3,4,5]

def get_images(num_images, dataset_dir):
    """ save num_images random images to dataset_dir """
    # first download labels
    request_url = LABELS_URL
    outfile = os.path.join(DATASET_DIR, data_utils.LABLES_FILENAME)
    
    # then download images
    for idx in range(num_images):
        img_id = str(random.randint(MIN_ID, MAX_ID)).zfill(6)
        perspective = str(random.choice(PERSPECTIVES))
        request_url = URL.format(img_id, perspective)
        print 'downloading image at: {}'.format(request_url)
        outfile = os.path.join(DATASET_DIR, '{}_{}.jpg'.format(img_id, perspective))
        urllib.urlretrieve(request_url, outfile)

def get_human_baseline(dataset_dir):
    # collect the filepaths in the directory    
    filepaths, filenames = data_utils.get_image_filepaths(dataset_dir)
    filepaths = np.random.permutation(filepaths)

    # load labels 
    labels = data_utils.load_labels_as_dict()

    # init counts 
    incorrect = 0
    correct = 0

    # iterate through images
    count = 0
    for filepath, filename in zip(filepaths, filenames):
        count += 1

        # get label
        img_id = data_utils.image_filename_to_id(filename)
        if img_id in labels:    
            label = labels[img_id]
        else:
            print 'cant find label'
            continue

        # load image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (227,227))

        # show image
        cv2.imshow('where was this picture taken? 1=ORL, 2=NYC, 3=PITT', img)

        # request a label until a valid one is given
        pred_label = None
        while pred_label not in [1,2,3]:

            try:
                pred_label = raw_input()
                if pred_label == 'q':
                    return
                pred_label = int(pred_label)
            except:
                print 'please enter 1 for Orlando, 2 for NYC, 3 for Pittsburg'

        if label != pred_label:
            incorrect += 1
        else:
            correct += 1

        # close image
        cv2.destroyWindow('where was this picture taken? 1=ORL, 2=NYC, 3=PITT')

        if count % 10 == 0:
            print 'current accuracy: {}'.format(correct / float(correct + incorrect))
            print 'current ratio: {} correct, {} incorrect'.format(correct, incorrect)

if __name__ == '__main__':
    num_images = 200
    dataset_dir = DATASET_DIR
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    get_images(num_images, dataset_dir)
    get_human_baseline(dataset_dir)
    shutil.rmtree(dataset_dir)
