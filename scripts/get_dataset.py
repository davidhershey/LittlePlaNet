
import math
import os
import random
import sys
from multiprocessing.pool import ThreadPool
import urllib

import file_utils

cities = {
"Paris" : (48.8567,2.3508),
"London" : (51.5072,-0.1275),
"Barcelona" : (41.3833,2.1833),
"Moscow" : (55.7500,37.6167),
"Sydney" : (-33.8650,151.2094),
"Rio" : (-22.9068,-43.1729),
"NYC" : (40.7127,-74.0059),
"SanFran" : (37.7833,-122.4167),
"Detroit" : (42.3314,-83.0458),
"DC" : (38.9047,-77.0164)
}

# radius of the Earth
R = 6378.1 

# radius of images around center of city
IMAGE_RADIUS = 10 

# number of images to download from each city
NUM_IMAGES_PER_CITY = 200

# size of failed-download image
FAILED_DOWNLOAD_IMAGE_SIZE = 3554

# place key in a file in the Geo-Localization directory 
# as the only text in the file on one line
KEY_FILEPATH = "../api_key.key"
API_KEY = file_utils.load_key(KEY_FILEPATH)
GOOGLE_URL = ("http://maps.googleapis.com/maps/api/streetview?sensor=false&"
              "size=256x256&fov=120&key=" + API_KEY)
IMAGES_DIR = '../imgs/'


def download_images_for_city(city, lat, lon):
    print 'downloading images of {}'.format(city)
    num_imgs = 0
    misses = 0
    cur_directory = os.path.join(IMAGES_DIR, city)
    if not os.path.exists(cur_directory):
        os.makedirs(cur_directory)

    while num_imgs < 100:
        
        # randomly select latitude and longitude in the city
        brng = math.radians(random.uniform(0, 360)) # bearing is 90 degrees converted to radians.
        d = random.uniform(0, IMAGE_RADIUS)
        lat_rad = math.radians(lat) # current lat point converted to radians
        lon_rad = math.radians(lon) # current long point converted to radians
        rand_lat = math.asin(math.sin(lat_rad)*math.cos(d/R) +
                        math.cos(lat_rad)*math.sin(d/R)*math.cos(brng))
        rand_lon = lon_rad + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat_rad),
                        math.cos(d/R)-math.sin(lat_rad)*math.sin(rand_lat))
        rand_lat = math.degrees(rand_lat)
        rand_lon = math.degrees(rand_lon)
        
        # download image
        filename = 'lat-{}-lon-{}.jpg'.format(round(rand_lat, 4), round(rand_lon, 4))
        filepath = os.path.join(cur_directory, filename)
        url = GOOGLE_URL + "&location=" + str(rand_lat) + ","+ str(rand_lon)
        urllib.urlretrieve(url, filepath)

        # check if the downloaded image was invalid and if so remove it
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            if size == FAILED_DOWNLOAD_IMAGE_SIZE:
                os.remove(filepath)
                misses += 1
            else:
                num_imgs += 1

    print 'invalid photo of {} downloaded {} times'.format(city, misses)
    file_utils.upload_directory_to_aws(cur_directory)

def download_images():

    # download images for each city in a different thread
    num_threads = 4
    pool = ThreadPool(num_threads)
    for city, (lat, lon) in cities.iteritems():
        pool.apply_async(download_images_for_city, (city, lat, lon))

    pool.close()
    pool.join()

if __name__ == '__main__':
    download_images()
