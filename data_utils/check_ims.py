import csv, json
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
from PIL import Image

root=Path('/mnt/data/omkumar/foundation_phase1/datasets/cataract1k/masks')
csvp=Path('/mnt/data/omkumar/foundation_phase1/data_utils/classes_in_patients.csv')
# read cases with num_classes >27
cases=set()
with csvp.open() as f:
    reader=csv.DictReader(f)
    for r in reader:
        try:
            if int(r['num_classes'])>27:
                cases.add(r['case_id'])
        except:
            pass

print('cases to process:', len(cases))

images_over_20=[]
case_distributions={}
# iterate
for ci,case in enumerate(sorted(cases)):
    case_dir=root/case
    if not case_dir.exists():
        # skip if missing
        continue
    counts=Counter()
    total=0
    # find image files (assume png/jpg/tif)
    for p in sorted(case_dir.rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}:
            try:
                im=np.array(Image.open(p))
                # if multi-channel, assume labels in single channel
                if im.ndim>2:
                    im=im[...,0]
                unique=np.unique(im)
                # ignore background 0 if present
                uniq_nonzero=[u for u in unique if u!=0]
                n=len(uniq_nonzero)
                counts[n]+=1
                total+=1
                if n>20:
                    images_over_20.append(str(p))
            except Exception as e:
                # skip unreadable
                #print('err',p,e)
                pass
    if total>0:
        # compute percentage distribution
        dist={str(k):round(v/total*100,3) for k,v in sorted(counts.items())}
        case_distributions[case]={'distribution_percent':dist,'total_images':total}
    # progress
    if (ci+1)%50==0:
        print('processed',ci+1,'cases')

# save json of images >20
out=Path('/mnt/data/omkumar/masks_images_over_20_classes.json')
with out.open('w') as f:
    json.dump(images_over_20,f,indent=2)

# save the per-case distributions
out2=Path('/mnt/data/omkumar/case_class_distributions.json')
with out2.open('w') as f:
    json.dump(case_distributions,f,indent=2)

print('done: images_over_20:',len(images_over_20))
print('saved to',out)
print('cases with distributions saved to',out2)