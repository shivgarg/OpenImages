import pandas as pd
import sys

bbox = pd.read_csv(sys.argv[1])
vrd = pd.read_csv(sys.argv[2])

vrd = vrd[vrd['RelationshipLabel']=='is']
vrd =vrd.drop(columns=['RelationshipLabel','XMin2','XMax2','YMin2','YMax2'])
bbox = bbox.drop(columns=['Source','Confidence','IsOccluded','IsTruncated','IsDepiction','IsInside','IsGroupOf','Label'])

vrd = vrd.rename(index=str,columns={'LabelName1':'LabelName','XMin1':'XMin','XMax1':'XMax','YMax1':'YMax','YMin1':'YMin'})

#join_result = bbox.set_index(['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']).join(vrd.set_index(['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']),how='outer')
attr_obj = set(vrd['LabelName'])
filtered_bbox=bbox.loc[bbox['LabelName'].isin(attr_obj)]
images = set(vrd['ImageID'])

filtered_bbox_neg=filtered_bbox.loc[~filtered_bbox['ImageID'].isin(images)]
filtered_bbox_pos=filtered_bbox.loc[filtered_bbox['ImageID'].isin(images)]

grouped_bbox = filtered_bbox_pos.groupby(['ImageID','LabelName'])
grouped_vrd = vrd.groupby(['ImageID','LabelName'])

count = 0
cnt = 0
print(len(filtered_bbox_pos))
print(len(vrd))
keys = grouped_vrd.groups.keys()
for name,rows in grouped_bbox:
    if name in keys:
        pos_samples = grouped_vrd.get_group(name)
    for row in rows.itertuples(index=False):
        if cnt%10000 == 0:
            print(cnt)
        cnt+=1
        is_positive = False
        if name in keys:
            for row1 in pos_samples.itertuples(index=False):
                xmin = max(row1[3],row[2])
                xmax = min(row1[4],row[3])
                if xmax < xmin :
                    continue
                ymax = min(row1[6],row[5])
                ymin = max(row1[5],row[4])
                if ymax < ymin:
                    continue
                area_inter = (ymax-ymin)*(xmax-xmin)
                area_union = (row1[4]-row1[3])*(row1[6]-row1[5]) + (row[3]-row[2])*(row[5]-row[4]) -area_inter
                if area_inter > 0.9*area_union:
                    is_positive = True
                    break
        if not is_positive:
            vrd.loc[len(vrd)] = list(row[:2] + ('nop',) + row[2:])
            count += 1
        
print(count)
print(len(vrd))
vrd.to_csv(sys.argv[3])
