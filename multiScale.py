import torchvision.transforms as T
import numpy as np
import torch


to_tensor= T.Compose([T.ToTensor()])
to_pil=T.Compose([T.ToPILImage()])

def image_resize(images):
  img=[to_tensor(to_pil(i).resize([512,512])) for i in images]
  img=[np.array(i) for i in img]
  img=np.array(img)
  img=torch.from_numpy(img)
  return img
