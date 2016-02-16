import os
import random
import sys
import urllib
import math
import getcolor
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
R = 6378.1 #Radius of the Earth

IMAGE_RADIUS = 10 #radius of images around center of city
API_KEY = "ENTER_API_KEY_HERE"
GOOGLE_URL = ("http://maps.googleapis.com/maps/api/streetview?sensor=false&"
              "size=640x640&fov=120&key=" + API_KEY)

for city in cities:
    print city
    num_imgs = 0
    misses=0
    while num_imgs < 200:
        if not os.path.exists("imgs/" + str(city)):
            os.makedirs("imgs/" + str(city))
        brng = math.radians(random.uniform(0,360)) #Bearing is 90 degrees converted to radians.
        d = float(random.uniform(0,IMAGE_RADIUS))
        lat1 = math.radians(cities[city][0]) #Current lat point converted to radians
        lon1 = math.radians(cities[city][1]) #Current long point converted to radians
        rand_lat = math.asin( math.sin(lat1)*math.cos(d/R) +
                                math.cos(lat1)*math.sin(d/R)*math.cos(brng))

        rand_lon = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
                                    math.cos(d/R)-math.sin(lat1)*math.sin(rand_lat))
        rand_lat = math.degrees(rand_lat)
        rand_lon = math.degrees(rand_lon)

        outfile = os.path.join("imgs/" + str(city),str(num_imgs) + ".jpg")
        url = GOOGLE_URL + "&location=" + str(rand_lat) + ","+ str(rand_lon)
        urllib.urlretrieve(url, outfile)
        if os.path.isfile(outfile):
            # get_color returns the main color of image
            color = getcolor.get_color(outfile)
            print color
            if color[0] == '#e3e2dd' or color[0] == "#e3e2de":
                misses+=1
                if misses%10 is 0:
                    print "misses: ",misses
                os.remove(outfile)
            else:
                num_imgs += 1
