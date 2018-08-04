import os
from PIL import Image
import sys

image_dir = sys.argv[1]
for _,_,files in os.walk(image_dir):
    for f in files:
        image = Image.open(image_dir+'/'+f)
        if image.mode != 'RGB':
            image.convert('RGB').save(image_dir+'/'+f)
            