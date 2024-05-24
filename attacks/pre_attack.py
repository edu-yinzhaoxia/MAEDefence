import os

# import foolbox
# import numpy as np
# import copy
# import shutil
# import time
# import imageio
# import math
# import torch
import torchattacks
# import torchvision
# import cv2
# from torch import nn
from torchvision.transforms import transforms

import utils
from utils.SINIFGSM import SINIM
from utils.VT import VMIM

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def FGSM(model, images, labels):
    attack = torchattacks.FGSM(model, eps=16 / 255)
    perturbed_images = attack(images, labels)
    return perturbed_images


def BIM(model, images, labels):
    attack = torchattacks.BIM(model, eps=8 / 255, alpha=1 / 255, steps=10)
    perturbed_images = attack(images, labels)
    return perturbed_images


def PGD(model, images, labels):
    attack = torchattacks.PGD(model, eps=16 / 255, alpha=1 / 255, steps=40, random_start=True)
    perturbed_images = attack(images, labels)
    return perturbed_images


def MIM(model, images, labels):
    attack = torchattacks.MIFGSM(model, eps=8 / 255, steps=10, decay=1.0)
    perturbed_images = attack(images, labels)
    return perturbed_images

def CW(model, images, labels):
    attack = torchattacks.CW(model, c=1, kappa=100, steps=500, lr=0.01)
    perturbed_images = attack(images, labels)
    return perturbed_images

def DIM(model, images, labels):
    attack = torchattacks.DIFGSM(model, eps=16/ 255, alpha=1/255, steps=20, decay=1.0, diversity_prob=0.7)
    perturbed_images = attack(images, labels)
    return perturbed_images


def TIM(model, images, labels):
    attack = torchattacks.TIFGSM(model, eps=16 / 255, alpha=1/255, steps=20, decay=1.0)
    perturbed_images = attack(images, labels)
    return perturbed_images

def SINIFGSM(model, images, labels):
    attack = torchattacks.SINIFGSM(model, eps=2 / 255,  alpha=1/255, steps=40, decay=1.0, m=5)
    perturbed_images = attack(images, labels)
    return perturbed_images

def VNI(model, images, labels):
    attack = torchattacks.VNIFGSM(model, eps=2 / 255, alpha=1/255, steps=10, decay=1.0, N=20)
    perturbed_images = attack(images, labels)
    return perturbed_images


def VMI(model, images, labels):
    attack = torchattacks.VMIFGSM(model, eps=16 / 255, alpha=1/255, steps=10, decay=1.0, N=20)
    perturbed_images = attack(images, labels)
    return perturbed_images


def deepfool(model, images, labels):
    attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
    perturbed_images = attack(images, labels)
    return perturbed_images
