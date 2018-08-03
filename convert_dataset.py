import pandas as pd 
import sys


bbox = pd.read_csv(sys.argv[1])
grouped = bbox.groupby('ImageID')

image_dir = sys.argv[2]
anno_dir = sys.argv[3]

for name,group in grouped:
    image_path = image_dir + '/' + name + '.jpg'
    anno_file = anno_dir + '/' + name + '.xml'
    