from sig import*
from utils import*
from meaning import*
import matplotlib.pyplot as plt
import cv2
import numpy as np
siggraph = SIGGRAPHGenerator()


path_to_model = "g_next.pth"
path_imag = "my_data/000000000785.jpg"
path_mask = "my_data/000000000785.jpg"

siggraph.load_state_dict(torch.load(path_to_model))
siggraph.eval()


L_img , _ = to_tensor(path_imag)
size = (L_img.shape[1] , L_img.shape[2])
_ , ab_mask = to_tensor(path_mask, size)

ab_mean = meaning(ab_mask, 15)
with torch.no_grad():
    ab_img = siggraph(torch.unsqueeze(L_img, dim = 0),
                    torch.unsqueeze(ab_mean, dim = 0))


img = lab_to_rgb(torch.unsqueeze(L_img, dim = 0), ab_img)
mask = lab_to_rgb(torch.unsqueeze(L_img, dim = 0), torch.unsqueeze(ab_mean, dim = 0))

img = 255*img[0]
img = img.astype('uint8')


mask = 255*mask[0]
mask = mask.astype('uint8')

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
cv2.imwrite('img.jpg', img)
cv2.imwrite('mask.jpg', mask)