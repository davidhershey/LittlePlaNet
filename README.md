# City-Recognition

This project is a smale-scale spin on Google DeepMind's PlaNet geolocalization Neural Network.  It was developed for CS231n at Stanford University.

Google's PlaNet: http://arxiv.org/abs/1602.05314

# Getting Started

## Downloading the dataset
- the dataset we used cannot be included in the repo, but we have provided the [script](https://github.com/dmakian/LittlePlaNet/blob/master/scripts/get_dataset.py) we used to download the data from google street view
  + you need a google street view account and key to use this script
  + create an account [here](https://developers.google.com/maps/documentation/streetview/)
  + after creating an account, get an api key, and store it in a file called "api_key.key" in the main repo directory
  + this will allow you to run the script and download the images to use as the dataset
  + the dataset downloads images from various cities that have been hard-coded in the `get_dataset.py` script, so change these coordinates if you'd like to focus on different cities

## Feature extraction
- this project used a pre-trained CNN to extract features for use with a linear classifier
  + for access to the trained models, see the [model repo](https://github.com/wulfebw/LittlePlaNet-Models)
- for extracting features see [`extract_features.py`](https://github.com/wulfebw/LittlePlaNet-Models/blob/master/scripts/extract_features.py)

## Model training
- after extracting features, you can use them with a variety of classification models
- see [`linear_classification.py`](https://github.com/dmakian/LittlePlaNet/blob/master/scripts/linear_classification.py) for an example script for training and evaluating some baseline models on the extracted features
