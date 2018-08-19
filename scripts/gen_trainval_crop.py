import pandas as pd 
import sys
import os
import xml.etree.cElementTree as ET
from PIL import Image
import random

bbox = pd.read_csv(sys.argv[1])
grouped = bbox[bbox['RelationshipLabel']!='is'].groupby('ImageID')


path = sys.argv[2]
trainval = open(path+'/trainval.txt','w')
test = open(path+'/test.txt','w')

arr = []
for name,group in grouped:
    cnt = 0
    for _,row in group.iterrows():
        cnt += 1
        arr.append(name+'_'+str(cnt))

random.shuffle(arr)
num_trainval = int(len(arr)*9/10)
for i in range(num_trainval):
    trainval.write(arr[i]+'\n')
for i in range(num_trainval,len(arr)):
    test.write(arr[i]+'\n')

trainval.close()
test.close()

