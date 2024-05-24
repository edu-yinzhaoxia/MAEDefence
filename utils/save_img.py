from torchvision import transforms
import numpy as np
import torch
import os


imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])
imagenet_std = torch.Tensor([0.229, 0.224, 0.225])

def save_adv_img(root_dir, imgs, save_dir ):
    filelist = os.listdir(root_dir)
    transform = transforms.Compose([transforms.ToPILImage(),
                                    ])
    for i, (filename) in enumerate(filelist):
        img = imgs[i]
        img = transform(img)
        imgpath = os.path.join(save_dir, filename)
        img.save(imgpath)



def save_reconstruct_img(root_dir, imgs, save_dir):
    filelist = os.listdir(root_dir)
    transform = transforms.Compose([transforms.ToPILImage(),
                                    ])
    for i, (filename) in enumerate(filelist):
        img = imgs[i]
        img = torch.Tensor.permute(img, (2, 0, 1))
        image = img.clone()
        for j in range(3):
            image[j] = image[j] * imagenet_std[j] + imagenet_mean[j]

        image = torch.clamp(image, min=0, max=1)
        image = transform(image)
        imgpath = os.path.join(save_dir, filename)
        image.save(imgpath)
