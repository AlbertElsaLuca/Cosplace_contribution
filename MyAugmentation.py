"""
In this file we will construct a couple of transformation that can be applied on an image 
for data augmentation in other to allow the programm to generalise better
"""
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision.transforms.transforms import ToPILImage
from torch.functional import Tensor
from torchvision.transforms import transforms
import torchvision.transforms as T
import augmentations

#Let's try some famous transformation that torch offers
#transform to tensor
T0=T.ToTensor() 
T1=T.RandomCrop(512,padding=4,padding_mode="reflect")
T2=T.RandomHorizontalFlip(p=0.6)
T3=T.RandomRotation(degrees=20,fill=150)   #rotation will randomly selected from -20 degree to 20 degree
T4=T.CenterCrop(size=(512,512))
T5=T.Grayscale(num_output_channels=3) #return grayscale image with R=G=B
T6=T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5) #already done by the professor
T7=T.RandomResizedCrop(512,scale=(0.5,0.9),ratio=(1,1))
T8=T.RandomInvert(p=0.3)
T9=T.RandomSolarize(threshold=0.75,p=0.3)
#second set of transformation base on RandomErase and RandomPerspective
T12= T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
T15= T.PILToTensor() # convert PIL image to tensor image
T20=T.ToPILImage()  # convert tensor to PIL images
T16=T.ConvertImageDtype(torch.float)
T_code=T.Compose([T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
                  T.RandomResizedCrop((512,512)),
                  T0,T16,
                  ])

#Random Erasing and RandonPerception
T13= T.RandomErasing(p=0.35)
T14= T.RandomPerspective(p=0.1,fill=150)
T10=T.RandomEqualize(p=0.5)
policies = [T.AutoAugmentPolicy.CIFAR10, T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]
T11=[T.AutoAugment(policy) for policy in policies]
T111=T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)
T112=T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)
Erasing = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                              transforms.RandomErasing(p=0.5),
                               ])
Erasing_norm=transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                transforms.RandomErasing(p=0.4),
                                 ])
T21=transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                 ])
"""
now using Torchvision.transforms.Compose() we can applied a chain of transformation to each
images before passing it to the training model

in order to maximise the training, we will pass a set of transformation to some epoch and another 
set to other epoch. for example for the first and third epoch we can applied the transformation 1 and 3 
and for epoch second ans fourth epoch apply transformation 2 4 and 5 in other to send 
many different perspective to the model.
"""
#given a list of images return a torch.size([B,C,H,W])
def transform_to_batch_Ofimages(images_augmented):
  augmented_images = [np.array(img.cpu()) for img in images_augmented]
  augmented_images=np.array(augmented_images)
  augmented_images_batch=torch.from_numpy(augmented_images)
  #print(type(augmented_images_batch))
  return augmented_images_batch   # torch.size([B,C,H,W])

#RandomHorizontalFlip + RandomRotation + ColorJitter + Totensor
def Transformation1(images):
  #images is a batch of images. so we should apply the set of transformation for each image in the batch
  trans=T.Compose([T2,T3,T6])
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images

#RandomHorizontalFlip + centerCrop +ColorJitter + Totensor
def Transformation2(images):
  trans=T.Compose([T2,T4,T6])
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images
#CenterCrop + RandomSolarize 
def Transformation3(images):
  trans=T.Compose([T4,T9])
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images

#to grayscale + RandomInvert
def Transformation4(images):
  trans=T.Compose([T5,T8])
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images


def Transformation5(images):
  trans=T.Compose([T20,T112,T0,Erasing]) #Autoaugment only accept either torch.int8 or PIL images
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images

#RandomPerspective
def Transformation6(images):
  trans=T.Compose([T14,T6])
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images

#RandomErasing
def Transformation7(images):
  trans=T.Compose([T14,T6])
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images
 #RandomCrop + RandomErasing
def Transformation8(images):
  trans=T.Compose([T7,Erasing])
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images

def Transformation9(images):
  trans=T.Compose([T20,T111,T0,Erasing]) #Autoaugment only accept either torch.int8 or PIL images
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images
#normalise the values
def Transformation10(images):
  trans=T.Compose([T21]) 
  augmented_images=[trans(img) for img in images]
  augmented_images=transform_to_batch_Ofimages(augmented_images)
  return augmented_images
def DefaultTransformation(images):
  gpu_augmentation = T.Compose([
            augmentations.DeviceAgnosticColorJitter(brightness=0.7,contrast=0.7,saturation=0.7,hue=0.3),                                      
            augmentations.DeviceAgnosticRandomResizedCrop([512, 512],scale=[1-0.5, 1]),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])                                                 
  augmented_images=gpu_augmentation(images)
  return augmented_images

#each of the above transformation takes as input a batch of tensor image 
#with tensor.size([B,C,H,W]) and return a batch or augmented_tensor images 
#with tensor.size([B,C,H,W])



