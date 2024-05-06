import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shapely

def png_to_grid(filename, width, height):
    im = np.array((Image.open(filename)).convert('L').resize((width,height)))#you can pass multiple arguments in single line
    #occ_array = np.full([width,height], 0)
    
    occ_array = 1.0 - ((im) / (255)) # Normalises and flips for occupancy as 255 is white, which isn't occupied.
    print(occ_array.shape)
    return occ_array
    #gr_im= Image.fromarray(im).save('gr_kolala.png')

#array1 = png_to_grid("ExampleMap.png", 200, 200)
#print(array1)

def grid_to_png(grid):
    occ_array = 1 - ((grid) * (255))
    im2 = Image.fromarray(grid)
    im2.save("ExampleResult.png")

def make_a_square(coords, size):
    a = coords[0]
    b = coords[1]
    return [(0 + a, 0 + b), (0 + a, size + b), (size + a, size + b), (size + a, 0 + b)]

def png_to_shapely_map(filename):
    #im = plt.imread(filename)
    #https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python

    img = cv2.imread(filename)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 125, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inners = []
    for contour in contours:
        current_shape = []
        for coord in contour:
            current_shape.append(tuple(coord[0]))
        #print(current_shape)
        inners.append(current_shape)
    trimmed_inners = [x for x in inners if len(x) >= 4]
        
    outers = make_a_square([0,0], 1000)
    map_area = shapely.Polygon(outers, trimmed_inners[1:]).simplify(0.5, preserve_topology=False)

    #map_area = shapely.reverse(map_area)

    bounding_box = shapely.Polygon(make_a_square([-10,-10], 1020))
    inverted_polygon = bounding_box.symmetric_difference(map_area)
    map_area = inverted_polygon
    return map_area