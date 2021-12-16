from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, optim
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import numpy as np


    
def to_tensor(path_to_image, size = (512,512)):
    img = Image.open(path_to_image).convert("RGB")

    # img = transforms.Resize(size, Image.BICUBIC)(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
    return L, ab



def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)