import argparse
import os

import numpy as np
import torch
import torchvision.transforms as tfs
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from metrics import psnr, ssim
from models.C2PNet import C2PNet

model_dir = 'trained_models/OTS.pkl'
net = C2PNet(gps=3, blocks=19)
ckp = torch.load(model_dir)
net.load_state_dict(ckp['model'])
net.eval()

haze_path = 'data/SOTS/outdoor/EW_dehaze_test.jpg'
output_path = 'data/SOTS/outdoor/dehaze/EW_dehaze_test.jpg'
model_dir = 'trained_models/OTS.pkl'
haze = Image.open(haze_path).convert('RGB')

# clear_im = im.split('_')[0] + '.png'
haze1 = tfs.ToTensor()(haze)[None, ::]
with torch.no_grad():
    pred = net(haze1)
ts = torch.squeeze(pred.clamp(0, 1))
vutils.save_image(ts, output_path)