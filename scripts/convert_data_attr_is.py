import pandas as pd 
import sys
import os
import xml.etree.cElementTree as ET
from PIL import Image


bbox = pd.read_csv(sys.argv[1])
grouped = bbox.groupby('ImageID')

image_dir = sys.argv[2]
anno_dir = sys.argv[3]
logfile = open(sys.argv[4],'w')

objs = {'nop':'0 ','/m/02gy9n':'1','/m/05z87':'2','/m/0dnr7':'3','/m/04lbp':'4','/m/083vt':'5'}
cnt = 0 
for name,group in grouped:
    image_path = image_dir + '/' + name + '.jpg'
    anno_file = anno_dir + '/' + name + '.xml'
    if not os.path.isfile(image_path):
        logfile.write(image_path+'\n')
        continue
    im = Image.open(image_path)
    root = ET.Element("annotation")
    ET.SubElement(root,"folder").text = 'OpenImages'
    ET.SubElement(root,"filename").text = name+'.jpg'
    ET.SubElement(root,"segmented").text = '0'
    size = ET.SubElement(root,"size")
    ET.SubElement(size,"width").text = str(im.width)
    ET.SubElement(size,"height").text = str(im.height)
    ET.SubElement(size,"depth").text = '3'
    for _,row in group.iterrows():
        cnt += 1
        obj = ET.SubElement(root,"object")
        ET.SubElement(obj,"name").text = row['LabelName']
        ET.SubElement(obj,"pose").text = 'Unspecified'
        ET.SubElement(obj,"difficult").text = '0'
        bndbox = ET.SubElement(obj,"bndbox")
        ET.SubElement(bndbox,"xmin").text = str(int(max(row['XMin'],0)*im.width))
        ET.SubElement(bndbox,"ymin").text = str(int(max(row['YMin'],0)*im.height))
        ET.SubElement(bndbox,"xmax").text = str(int(min(row['XMax'],1)*im.width))
        ET.SubElement(bndbox,"ymax").text = str(int(min(row['YMax'],1)*im.height))
        ET.SubElement(bndbox,"label").text = objs[row['LabelName2']]
    tree = ET.ElementTree(root)
    print("Writing to file ", anno_file, cnt)
    tree.write(anno_file)
            
logfile.close()
