import os
from PIL import Image
import sys
from multiprocessing import Pool

image_dir = sys.argv[1]

def convert(img):
    image = Image.open(img)
    if image.mode != 'RGB':
        print(img)
        image.convert('RGB').save(img)

for folder,_,files in os.walk(image_dir):
  print(folder)
  file_list = [folder + '/' + f for f in files]
  with Pool(8) as p:
        p.map(convert,file_list)
