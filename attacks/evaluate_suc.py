import torch.nn as nn
import os
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm

# from utils.JPEG import jpeg
from utils.load_img import load_imagenet_val

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from utils.auxiliary_utils import *
from utils.auxiliary_utils import attack_success
import torchvision.models as models

# pretrained=True就可以使用预训练的模型
or_img_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\val_rs'
adv_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\adversarial_img'
denoise_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\denoise_img'
label_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\val_rs.xls'
re_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\re_outputs'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
device = torch.device("cuda")
resnet50 = nn.Sequential(
        normalize,
        models.resnet50(pretrained=True))
resnet101 = nn.Sequential(
        normalize,
        models.resnet101(pretrained=True))
vgg16 = nn.Sequential(
        normalize,
        models.vgg16(pretrained=True))
vgg19 = nn.Sequential(
        normalize,
        models.vgg19(pretrained=True))
densenet121 = nn.Sequential(
        normalize,
        models.densenet121(pretrained=True))
densenet201 = nn.Sequential(
        models.densenet201(pretrained=True))
inc_v3 = nn.Sequential(
        torchvision.transforms.Resize(299),
        normalize,
        models.inception_v3(aux_logits=False, pretrained=True))
seed = 18
if seed != -1:
    print('set seed : ', seed)
    setup_seed(seed)
resnet50 = resnet50.to(device)
resnet101 = resnet101.to(device)
VGG16 = vgg16.to(device)
VGG19 = vgg19.to(device)
densenet121 = densenet121.to(device)
densenet201 = densenet201.to(device)
inc_v3 = inc_v3.to(device)

# jpeg压缩
# jpeg(adv_dir, re_dir)
data, num_images = load_imagenet_val(re_dir, label_dir)



suc_img0 = 0
suc_img1 = 0
suc_img2 = 0
suc_img3 = 0
suc_img4 = 0
suc_img5 = 0
suc_img6 = 0

total_img = 1000

pbar = tqdm(total=1000 / 50)
for batch, (inputs, targets) in enumerate(data):
    # 镜像翻转
    # inputs = inputs.flip(dims=[3])

    inputs = inputs.to(device)
    targets = targets.to(device)

    # attack and calculate ASR
    suc_id0 = attack_success(targets, predict(resnet50, inputs))
    suc_img0 += len(suc_id0)

    suc_id1 = attack_success(targets, predict(resnet101, inputs))
    suc_img1 += len(suc_id1)

    suc_id2 = attack_success(targets, predict(vgg16, inputs))
    suc_img2 += len(suc_id2)

    suc_id3 = attack_success(targets, predict(vgg19, inputs))
    suc_img3 += len(suc_id3)

    suc_id4 = attack_success(targets, predict(densenet121, inputs))
    suc_img4 += len(suc_id4)

    suc_id5 = attack_success(targets, predict(densenet201, inputs))
    suc_img5 += len(suc_id5)

    suc_id6 = attack_success(targets, predict(inc_v3, inputs))
    suc_img6 += len(suc_id6)
    pbar.update(1)


print("suc_resnet50: {:.2f}% ", 100.0 * suc_img0 / total_img)
print("suc_resnet101: {:.2f}%", 100.0 * suc_img1 / total_img)
print("suc_vgg16: {:.2f}%", 100.0 * suc_img2 / total_img)
print("suc_vgg19: {:.2f}%", 100.0 * suc_img3 / total_img)
print("suc_densenet121: {:.2f}% ", 100.0 * suc_img4 / total_img)
print("suc_densenet201: {:.2f}% ", 100.0 * suc_img5 / total_img)
print("suc_inc_v3: {:.2f}% ", 100.0 * suc_img6 / total_img)

