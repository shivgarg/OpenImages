import pandas as pd 
import sys
import os
import xml.etree.cElementTree as ET
from PIL import Image


bbox = pd.read_csv(sys.argv[1])
grouped = bbox[bbox['RelationshipLabel']!='is'].groupby('ImageID')

image_dir = sys.argv[2]
anno_dir = sys.argv[3]
logfile = open(sys.argv[4],'w')
objs = {'at':'0', 'on':'1', 'holds': '2', 'plays':'3', 'interacts_with':'4', 'wears':'5', 'inside_of':'6', 'under':'7', 'hits':'8'}

cnts = 0
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
    cnt = 0
    for _,row in group.iterrows():
        cnt += 1
        cnts += 1
        obj = ET.SubElement(root,"object")
        ET.SubElement(obj,"name").text = 'box'
        ET.SubElement(obj,"pose").text = 'Unspecified'
        ET.SubElement(obj,"difficult").text = '0'
        bndbox = ET.SubElement(obj,"bndbox")
        xmin = int(max(min(row['XMin1'],row['XMin2']),0)*im.width)
        xmax = int(min(max(row['XMax1'],row['XMax2']),1)*im.width)
        ymin = int(max(min(row['YMin1'],row['YMin2']),0)*im.height)
        ymax = int(min(max(row['YMax1'],row['YMax2']),1)*im.height)
        ET.SubElement(bndbox,"xmin").text = str(xmin)
        ET.SubElement(bndbox,"ymin").text = str(ymin)
        ET.SubElement(bndbox,"xmax").text = str(xmax)
        ET.SubElement(bndbox,"ymax").text = str(ymax)
        
        im.crop((xmin,ymin,xmax,ymax)).save(image_dir+'_crop/'+name+'_'+str(cnt)+'.jpg')
    
        root_crop = ET.Element("annotation")
        ET.SubElement(root_crop,"folder").text = 'OpenImages'
        ET.SubElement(root_crop,"filename").text = name+'_'+str(cnt)+'.jpg'
        ET.SubElement(root_crop,"segmented").text = '0'
        ET.SubElement(root_crop,"label").text = objs[row['RelationshipLabel']]
        size = ET.SubElement(root_crop,"size")
        ET.SubElement(size,"width").text = str(xmax-xmin)
        ET.SubElement(size,"height").text = str(ymax-ymin)
        ET.SubElement(size,"depth").text = '3'

        obj = ET.SubElement(root_crop,"object")
        ET.SubElement(obj,"name").text = row['LabelName1']
        ET.SubElement(obj,"pose").text = 'Unspecified'
        ET.SubElement(obj,"difficult").text = '0'
        bndbox = ET.SubElement(obj,"bndbox")
        xmin1 = max(max(row['XMin1'],0)*im.width - xmin,0)
        xmax1 = min(min(row['XMax1'],1)*im.width - xmin,xmax-xmin)
        ymin1 = max(max(row['YMin1'],0)*im.height - ymin,0)
        ymax1 = min(min(row['YMax1'],1)*im.height - ymin,ymax-ymin)
        ET.SubElement(bndbox,"xmin").text = str(int(xmin1))
        ET.SubElement(bndbox,"ymin").text = str(int(ymin1))
        ET.SubElement(bndbox,"xmax").text = str(int(xmax1))
        ET.SubElement(bndbox,"ymax").text = str(int(ymax1))

        obj = ET.SubElement(root_crop,"object")
        ET.SubElement(obj,"name").text = row['LabelName2']
        ET.SubElement(obj,"pose").text = 'Unspecified'
        ET.SubElement(obj,"difficult").text = '0'
        bndbox = ET.SubElement(obj,"bndbox")
        xmin1 = max(max(row['XMin2'],0)*im.width - xmin,0)
        xmax1 = min(min(row['XMax2'],1)*im.width - xmin,xmax-xmin)
        ymin1 = max(max(row['YMin2'],0)*im.height - ymin,0)
        ymax1 = min(min(row['YMax2'],1)*im.height - ymin,ymax-ymin)
        ET.SubElement(bndbox,"xmin").text = str(int(xmin1))
        ET.SubElement(bndbox,"ymin").text = str(int(ymin1))
        ET.SubElement(bndbox,"xmax").text = str(int(xmax1))
        ET.SubElement(bndbox,"ymax").text = str(int(ymax1))
        tree_crop = ET.ElementTree(root_crop)
        tree_crop.write(anno_dir+'_crop/'+name+'_'+str(cnt)+'.xml')
        print("Writing cropped anno file ",anno_dir+'_crop/'+name+'_'+str(cnt)+'.xml')
    tree = ET.ElementTree(root)
    print("Writing to file ", anno_file)
    print(cnts)
    tree.write(anno_file)
            
print(objs)
logfile.close()
