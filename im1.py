import skimage
from skimage import io,color,transform
import matplotlib
from matplotlib import pyplot as plt
import math
import numpy as np

import os
import json

#Loads Data 
with open(r'C:\Users\Admin\Desktop\TKinter\annotations_trainval2017\annotations\instances_val2017.json') as f:
    data = json.load(f)


annotations = data['annotations']
image_data = data['images']

#List of image parameters and properties
image_parameters = []
for img in image_data:
    h = img['height']
    w = img['width']
    i = img['id']
    z = {'height':h,'width':w,'id':i}
    image_parameters.append(z)
 
# ist of object parameters
image_bbox = []
for img in annotations:
    i = img['image_id']
    c = img['category_id']
    x = img['bbox'][0]
    y = img['bbox'][1]
    w = img['bbox'][2]
    h = img['bbox'][3]
    m = {'id':i,'class':c,'x':x,'y':y,'w':w,'h':h}
    if m['class']==1:
        image_bbox.append(m)


#Creates the final list and normalizes the image data
final = []

for i in image_parameters:
    z = {'id':i['id'],'para':[]}
    for j in image_bbox:
        if i['id'] == j['id']:
            k = {'x':j['x']/i['width'],'y':j['y']/i['height'],'w':j['w']/i['width'],'h':j['h']/i['height'],'class':j['class']} 
            z['para'].append(k)
    final.append(z)


#Isolates Objects from desired object class
finalList = []
for i in final:
    for j in i['para']:
        if j['class'] == 1:
            check = True
    if check:       
        finalList.append(i)
    check = False
    

def id2name(Id):
    a = "0"*(12-len(Id))
    return a+Id+".jpg"
    
print(id2name('123456'))


def generateGrid(Input):
    grid = np.zeros((7,7,5))
    
    boxes = Input['para']
    
    for box in boxes:
        a = box['x']
        b = box['y']
        w = box['w']
        h = box['h']
        
        x = a+w/2
        y = b+h/2
        X = math.floor(x*7)
        Y = math.floor(y*7)
        
        grid[Y][X][0] = 1
        grid[Y][X][1] = x
        grid[Y][X][2] = y
        grid[Y][X][3] = math.sqrt(w)
        grid[Y][X][4] = math.sqrt(h)
    
    return {'name':id2name(str(Input['id'])),'grid':np.ndarray.tolist(grid)}  
    
Grid = []    
for item in finalList:
    Grid.append(generateGrid(item))


with open("data.json", "w") as outfile: 
    json.dump(Grid, outfile) 