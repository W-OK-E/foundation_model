import os
import tqdm
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir",type=str)

args = parser.parse_args()

for file in tqdm.tqdm(os.listdir(args.dir)):
    t_file = os.path.join(args.dir,file)
    per_t = torch.load(t_file).permute(2,0,1)
    torch.save(per_t,t_file)