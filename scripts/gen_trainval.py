import pandas as pd
import sys

vrd = pd.read_csv(sys.argv[1])
dataset_split = sys.argv[2]

vrd = vrd.sort_values(['ImageID'])
if dataset_split == 'is':
    vrd = vrd[vrd['RelationshipLabel'] == 'is'].groupby(['LabelName1','LabelName2'])
elif dataset_split == 'region':
    vrd = vrd[vrd['RelationshipLabel'] != 'is'].groupby(['RelationshipLabel'])
else:
    print("Use gen_trainval_crop.py\n")
    sys.exit(0)


path = sys.argv[3]
trainval = open(path+'/trainval.txt','w')
test = open(path+'/test.txt','w')

test_set = set()
trainval_set = set()
for name,rows in vrd:
    l = int(len(rows)/10)
    for i in range(l):
        test_set.add(rows.iloc[i]['ImageID']+'\n')
    for i in range(l,len(rows)):
        trainval_set.add(rows.iloc[i]['ImageID']+'\n')

for f in test_set:
    test.write(f)
for f in trainval_set:
    trainval.write(f)
trainval.close()
test.close()




