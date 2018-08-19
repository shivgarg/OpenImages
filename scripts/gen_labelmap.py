import pandas as pd
import sys

vrd = pd.read_csv(sys.argv[1])
dataset_split = sys.argv[2]

if dataset_split == "is":
    vrd = vrd[vrd['RelationshipLabel'] == 'is'].groupby(['LabelName1','LabelName2']).groups.keys()
elif dataset_split == "region":
    vrd = vrd[vrd['RelationshipLabel'] != 'is'].groupby('RelationshipLabel').groups.keys()
else:
    vrd1 = vrd[vrd['RelationshipLabel'] != 'is'].groupby('LabelName2').groups.keys()
    vrd2 = vrd[vrd['RelationshipLabel'] != 'is'].groupby('LabelName1').groups.keys()
    vrd = set(list(vrd1) + list(vrd2))

def gen_str(name,id):
    return "item { \n name : \""+name+"\"\n label: "+str(id)+ "\n display_name: \""+ name+"\"\n}\n"

f = open(sys.argv[3],'w')
f.write(gen_str('background',0))
i=1
for keys in vrd:
    f.write(gen_str(keys,i))
    i+=1
f.close()