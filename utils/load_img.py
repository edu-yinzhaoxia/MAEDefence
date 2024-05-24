import torchvision
from torchvision import datasets, transforms
import numpy as np
import random
import torch
import os
import shutil
from PIL import Image
import xlrd



imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def load_imagenet_val(img_dir, label_dir):
    def get_data(filename, sheetnum, label_dir):
        dir_case = label_dir
        data = xlrd.open_workbook(dir_case)
        table = data.sheets()[sheetnum]
        nor = table.nrows
        dict = {}
        for i in range(1, nor):
            title = table.cell_value(i, 0)  # 得到excle中第一列图片名作为键
            value = table.cell_value(i, 1)  # 得到excle中第二列标签作为值
            dict[title] = str(int(value))
            yield dict

    root_dir = img_dir
    filelist = os.listdir(root_dir)
    sub_Imagenet = []
    dict = {}
    for i in get_data('add_user', 0, label_dir):
        dict = i
    for j, (filename) in enumerate(filelist):
        label = dict[filename]
        label = int(label)-1
        imgpath = os.path.join(root_dir, filename)
        imge = Image.open(imgpath)
        imge = np.array(imge)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(299),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()
                                        ])
        imge = transform(imge)
        sub_Imagenet.append((imge, label))
    dataloader = torch.utils.data.DataLoader(dataset=sub_Imagenet,
                                             batch_size=50,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True
                                             )
    return dataloader, len(sub_Imagenet)




def load_adv_img(adv_img_dir):
    root_dir = adv_img_dir
    filelist = os.listdir(root_dir)
    sub_Imagenet = []
    for j, (filename) in enumerate(filelist):
        imgpath = os.path.join(root_dir, filename)
        img = Image.open(imgpath)
        img = img.resize((224, 224))
        img = np.array(img) / 255.
        img = img - imagenet_mean
        img = img / imagenet_std
        sub_Imagenet.append((img))
    dataloader = torch.utils.data.DataLoader(dataset=sub_Imagenet,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True
                                             )
    return dataloader, len(sub_Imagenet)


# def load_imagenet(img_dir, reimg_dir, label_dir):
#     def get_data(filename, sheetnum, label_dir):
#         dir_case = label_dir
#         data = xlrd.open_workbook(dir_case)
#         table = data.sheets()[sheetnum]
#         nor = table.nrows
#         dict = {}
#         for i in range(1, nor):
#             title = table.cell_value(i, 0)  # 得到excle中第一列图片名作为键
#             value = table.cell_value(i, 1)  # 得到excle中第二列标签作为值
#             dict[title] = str(int(value))
#             yield dict
#
#     root_dir1 = img_dir
#     root_dir2 = reimg_dir
#     filelist = os.listdir(root_dir1)
#     sub_Imagenet = []
#     dict = {}
#     for i in get_data('add_user', 0, label_dir):
#         dict = i
#     for j, (filename) in enumerate(filelist):
#         label = dict[filename]
#         label = int(label)-1
#         imgpath1 = os.path.join(root_dir1, filename)
#         imgpath2 = os.path.join(root_dir2, filename)
#         imge1 = Image.open(imgpath1)
#         imge1 = np.array(imge1)
#         imge2 = Image.open(imgpath2)
#         imge2 = np.array(imge2)
#
#         transform = transforms.Compose([transforms.ToPILImage(),
#                                         transforms.Resize(299),
#                                         transforms.CenterCrop(224),
#                                         transforms.ToTensor()
#                                         ])
#         imge1 = transform(imge1)
#         imge2 = transform(imge2)
#         imge = (imge1+imge2)/2
#         sub_Imagenet.append((imge, label))
#     dataloader = torch.utils.data.DataLoader(dataset=sub_Imagenet,
#                                              batch_size=50,
#                                              shuffle=False,
#                                              num_workers=0,
#                                              pin_memory=True
#                                              )
#     return dataloader, len(sub_Imagenet)