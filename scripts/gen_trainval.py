import pandas as pd
import sys

vrd = pd.read_csv(sys.argv[1])
vrd = vrd[vrd['RelationshipLabel'] != 'is'].groupby(['RelationshipLabel'])

path = sys.argv[2]
trainval = open(path+'/trainval.txt','w')
test = open(path+'/test.txt','w')

for name,rows in vrd:
    l = int(len(rows)/10)
    for i in range(l):
        test.write(rows.iloc[i]['ImageID']+'\n')
    for i in range(l,len(rows)):
        trainval.write(rows.iloc[i]['ImageID']+'\n')

trainval.close()
test.close()




