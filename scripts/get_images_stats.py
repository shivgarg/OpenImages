import os
import argparse
import matplotlib.pyplot as plt 
from PIL import Image
args = argparse.ArgumentParser()
args.add_argument("image_dir")
args = args.parse_args()

size = []
aspect_ratio = []

for f,_,files in os.walk(args.image_dir):
    for fil in files:
        img = Image.open(os.path.join(f,fil))
        width = img.width
        height = img.height
        size.append(width*height)
        aspect_ratio.append(height/width)


plt.scatter(size,aspect_ratio)
plt.show()

