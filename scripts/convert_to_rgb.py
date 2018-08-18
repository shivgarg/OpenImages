import os
from PIL import Image
import sys
from multiprocessing import Pool

image_dir = sys.argv[1]

def convert(img):
    image = Image.open(image_dir+'/'+img)
    if image.mode != 'RGB':
        print(img)
        image.convert('RGB').save(image_dir+'/'+img)

for _,_,files in os.walk(image_dir):
    with Pool(4) as p:
        p.map(convert,files)
