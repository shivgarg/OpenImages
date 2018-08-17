import pandas as pd
import sys

vrd = pd.read_csv(sys.argv[1])

vrd = vrd[vrd['RelationshipLabel'] != 'is'].groupby('RelationshipLabel').groups.keys()


def gen_str(name,id):
    return "item { \n name : \""+name+"\"\n label: "+str(id)+ "\n display_name: \""+ name+"\"\n}\n"

f = open(sys.argv[2],'w')
f.write(gen_str('background',0))
i=1
for keys in vrd:
    f.write(gen_str(keys,i))
    i+=1
f.close()