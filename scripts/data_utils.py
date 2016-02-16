"""
:description: Utilities for loading in, preprocessing, and organizing data.
"""
import cv2
import glob
import numpy as np
import os
import scipy.io

# filename and directory constants
DATA_DIR = '../data'
CRCV_DIR = '/crcv'
PREPROCESSED_CRCV_DIR = '/preprocessed_crcv'
OUTPUT_DIR = '../outputs'
LABLES_FILENAME = 'GPS_Long_Lat_Compass.mat'
HIST_FILENAME = 'Color_hist.mat'
GIST_FILENAME = 'GIST.mat'

# data split constants
TRAIN_RATIO = .8
VAL_RATIO = .1
TEST_RATIO = .1

# feature constants
NUM_HIST_FEATURES = 60
NUM_HIST_ROWS_PER_LOCATION = 5

# preprocessing constants
IMAGE_SIDE_LENGTH = 256
NUM_IMAGE_CHANNELS = 3

# latitude and longitude min and max for each city
ORLANDO_LAT_MIN = 28
ORLANDO_LAT_MAX = 29
ORLANDO_LONG_MIN = -82
ORLANDO_LONG_MAX = -81
NYC_LAT_MIN = 40
NYC_LAT_MAX = 42
NYC_LONG_MIN = -75
NYC_LONG_MAX = -73
PITT_LAT_MIN = 40
PITT_LAT_MAX = 42
PITT_LONG_MIN = -81
PITT_LONG_MAX = -79

def load_mat(filename):
    """
    :description: Load and return the dictionary in a matlab file
    """
    assert filename.endswith('.mat'), 'load_mat requires .mat file'
    arr = scipy.io.loadmat(os.path.join(DATA_DIR, filename))
    return arr

def load_labels_as_dict():
    """
    :description: Loads labels for each row in the dataset. Labels returned
        as a dict where row number is the key and the label is the value, which 
        is one of {1 = orlando, 2 = nyc, 3 = pitt}.
    """
    gps = load_mat(LABLES_FILENAME)['GPS_Compass']
   
    # create mapping of <row number, label>
    # 1 = orlando, 2 = nyc, 3 = pitt
    labels = dict()
    for idx, (latitude, longitude, angle) in enumerate(gps):

        # orlando
        if latitude > ORLANDO_LAT_MIN and latitude < ORLANDO_LAT_MAX \
                and longitude > ORLANDO_LONG_MIN and longitude < ORLANDO_LONG_MAX:
            labels[idx] = 1

        # nyc
        elif latitude > NYC_LAT_MIN and latitude < NYC_LAT_MAX \
                and longitude > NYC_LONG_MIN and longitude < NYC_LONG_MAX:
            labels[idx] = 2

        # pitt
        elif latitude > PITT_LAT_MIN and latitude < PITT_LAT_MAX \
                and longitude > PITT_LONG_MIN and longitude < PITT_LONG_MAX:
            labels[idx] = 3

        # invalid latitude or longitude
        else:
            raise ValueError('Latitude or longitude not in valid range.\
                               Longitude: {}\tLatitude: {}'.format(latitude, longitude))

    return labels

def load_labels_as_list():
    """
    :description: Loads labels for each row in the dataset. Labels returned
        as a list of values where the index of the value is the XXXXXXX row
        of the corresponding image and the label itself is one of
        {1 = orlando, 2 = nyc, 3 = pitt}.
    """
    gps = load_mat(LABLES_FILENAME)['GPS_Compass']
   
    # collect the row numbers that correspond to each city
    # 1 = orlando, 2 = nyc, 3 = pitt
    labels = []
    for idx, (latitude, longitude, angle) in enumerate(gps):

        # orlando
        if latitude > ORLANDO_LAT_MIN and latitude < ORLANDO_LAT_MAX \
                and longitude > ORLANDO_LONG_MIN and longitude < ORLANDO_LONG_MAX:
            labels.append((idx, 1))

        # nyc
        elif latitude > NYC_LAT_MIN and latitude < NYC_LAT_MAX \
                and longitude > NYC_LONG_MIN and longitude < NYC_LONG_MAX:
            labels.append((idx, 2))

        # pitt
        elif latitude > PITT_LAT_MIN and latitude < PITT_LAT_MAX \
                and longitude > PITT_LONG_MIN and longitude < PITT_LONG_MAX:
            labels.append((idx, 3))

        # invalid latitude or longitude
        else:
            raise ValueError('Latitude or longitude not in valid range.\
                               Longitude: {}\tLatitude: {}'.format(latitude, longitude))

    # return the label corresponding to each row
    return np.array(labels)[:, 1]

def load_features_and_labels(feature_set='all'):
    """
    :description: load features and corresponding labels

    :type feature_set: string
    :param feature_set: Which features to load. One of {'all', 'hist', 'gist'}
    """
    
    # load the labels with the features so we can randomly permute the order
    labels = load_labels()

    # repeat the labels for each location
    labels = np.tile(labels, (NUM_HIST_ROWS_PER_LOCATION, 1)).T.reshape(-1)
        
    # load features
    if feature_set == 'all':
        hist = load_mat(HIST_FILENAME)['color_hist']
        gist = load_mat(GIST_FILENAME)['RefImGIST']
        feats = np.hstack((hist, gist))
    elif feature_set == 'hist':
        feats = load_mat(HIST_FILENAME)['color_hist']
    elif feature_set == 'gist':
        feats = load_mat(GIST_FILENAME)['RefImGIST']
    else:
        raise ValueError("""feature_set must be one of {'all', 'hist', 'gist'}""")
    
    # remove all zero rows - rows with data are separated by 5 all zero
    # rows, i'm not sure why but remove them here
    feats = feats[np.any(feats != 0, axis=1)]

    # permute features and labels
    # we want the test set to always be the same images, so set random seed before permuting
    np.random.seed(42)
    idxs = np.random.permutation(len(feats))
    feats = feats[idxs]
    labels = labels[idxs]

    # train, val, test split
    train_idx = int(len(feats) * TRAIN_RATIO)
    val_idx = int(len(feats) * (TRAIN_RATIO + VAL_RATIO))
    train = feats[:train_idx]
    y_train = labels[:train_idx]
    val = feats[train_idx: val_idx]
    y_val = labels[train_idx: val_idx]
    test = feats[val_idx:]
    y_test = labels[val_idx:] 
        
    # compute mean and std on train data
    means = np.mean(train, axis=0)
    stds = np.std(train, axis=0)

    # and use those stats to normalize train, val, test
    train = (train - means) / stds
    val = (val - means) / stds
    test = (test - means) / stds

    return train, y_train, val, y_val, test, y_test

def subtract_mean_image(imgs):
    mean_rgb_values = imgs.mean(axis=(0,1,2))
    imgs -= mean_rgb_values
    return imgs

def resize_image(img, side_length):
    """
    :description: Resize an image proportionally.
    """
    # resize proportionally and then crop to avoid warping image
    h, w, c = img.shape
    new_h, new_w = side_length, side_length
    if h > w:
        new_h = side_length * int(h / float(w))
    else:
        new_w = side_length * int(w / float(h))

    # crop center
    img = cv2.resize(img, (new_h, new_w))
    h_offset = (new_h - side_length) / 2
    w_offset = (new_w - side_length) / 2
    img = img[h_offset:h_offset + side_length, w_offset:w_offset + side_length]
    return img
    
def write_images_to_directory(output_dir, imgs, filenames):
    for img, filename in zip(imgs, filenames):
        if filename is not None:
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)

def image_filename_to_id(filename):
    return int(filename[: filename.rindex('_')])
            
def get_image_filepaths(dataset_dir):
    pattern = os.path.join(dataset_dir, '*.jpg')
    filepaths = glob.glob(pattern)
    filenames = [filepath[filepath.rindex('/') + 1:] for filepath in filepaths]
    return filepaths, filenames
    
def preprocess_crcv(dataset_dir, output_dir):
    # get filepaths of images to process and their corresponding ids
    filepaths, img_filenames = get_image_filepaths(dataset_dir)
    num_imgs = len(filepaths)

    # allocate containers for processed images and filenames
    imgs = np.empty((num_imgs, IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH, NUM_IMAGE_CHANNELS))
    filenames = [None] * num_imgs
    
    # load and resize each image
    for idx, (filepath, filename) in enumerate(zip(filepaths, img_filenames)):
        print('processing filepath: {}'.format(filepath))
        img = cv2.imread(filepath)
        if img is None:
            print('image filepath failed to load: {}'.format(filepath))
        else:
            resized_img = resize_image(img, IMAGE_SIDE_LENGTH)
            imgs[idx] = resized_img
            filenames[idx] = filename

    # subtract mean image from images
    imgs = subtract_mean_image(imgs)
            
    # write the preprocessed images to dir
    write_images_to_directory(output_dir, imgs, filenames)

def write_labels_file(dataset_dir, output_filepath):
    # collect the filepaths in the directory    
    filepaths, filenames = get_image_filepaths(dataset_dir)

    # load labels 
    labels = load_labels_as_dict()

    # format the lines for each data entry
    lines = []
    for filepath, filename in zip(filepaths, filenames):
        img_id = image_filename_to_id(filename)
        if img_id in labels:    
            label = labels[img_id]
        else:
            print 'cant find label'
        string = '{} {}\n'.format(filename, label)
        lines.append(string)

    # write the lines to file
    with open(output_filepath, 'wb') as f:
        f.writelines(lines)
                        
if __name__ == '__main__':
    # dataset_dir = os.path.join(DATA_DIR, 'small_dataset')
    # output_dir = os.path.join(OUTPUT_DIR, 'small_dataset')
    # preprocess_crcv(dataset_dir, output_dir)

    train_dir = os.path.join(OUTPUT_DIR, 'small_dataset/train')
    train_data_file = os.path.join(OUTPUT_DIR, 'small_dataset/train.txt')
    write_labels_file(train_dir, train_data_file)

    val_dir = os.path.join(OUTPUT_DIR, 'small_dataset/val')
    val_data_file = os.path.join(OUTPUT_DIR, 'small_dataset/val.txt')
    write_labels_file(val_dir, val_data_file)
    
