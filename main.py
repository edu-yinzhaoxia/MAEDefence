import sys
import os
from torch import nn
from torchvision import transforms
import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
from attacks.attack import attack_img
from utils.auxiliary_utils import attack_success, predict
from utils.denoise import wavelet_denoise
from utils.load_img import load_adv_img, load_imagenet_val
from utils.save_img import save_reconstruct_img, save_adv_img
sys.path.append('./reconstruction_image/mae-main')
import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
device = torch.device("cuda")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
re_imgs = []
ge_imgs = []
or_img_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\val_rs'
adv_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\adversarial_img'
denoise_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\Re_image'
label_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\val_rs.xls'
save_dir1 = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\re_outputs'




def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=24)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model,mask_ratio):
    x = torch.as_tensor(img)
    # make it a batch-like
    # x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    if mask_ratio==0.25:
        loss1, y1, mask1, loss2, y2, mask2, loss3, y3, mask3, loss4, y4, mask4 = model(x.float(), mask_ratio)
        y1 = model.unpatchify(y1)
        y1 = torch.einsum('nchw->nhwc', y1).detach().cpu()

        y2 = model.unpatchify(y2)
        y2 = torch.einsum('nchw->nhwc', y2).detach().cpu()

        y3 = model.unpatchify(y3)
        y3 = torch.einsum('nchw->nhwc', y3).detach().cpu()

        y4 = model.unpatchify(y4)
        y4 = torch.einsum('nchw->nhwc', y4).detach().cpu()
        # visualize the mask
        mask1 = mask1.detach()
        mask1 = mask1.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask1 = model.unpatchify(mask1)  # 1 is removing, 0 is keeping
        mask1 = torch.einsum('nchw->nhwc', mask1).detach().cpu()

        mask2 = mask2.detach()
        mask2 = mask2.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask2 = model.unpatchify(mask2)  # 1 is removing, 0 is keeping
        mask2 = torch.einsum('nchw->nhwc', mask2).detach().cpu()

        mask3 = mask3.detach()
        mask3 = mask3.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask3 = model.unpatchify(mask3)  # 1 is removing, 0 is keeping
        mask3 = torch.einsum('nchw->nhwc', mask3).detach().cpu()

        mask4 = mask4.detach()
        mask4 = mask4.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask4 = model.unpatchify(mask4)  # 1 is removing, 0 is keeping
        mask4 = torch.einsum('nchw->nhwc', mask4).detach().cpu()
        x = torch.einsum('nchw->nhwc', x)

        # masked imagea
        im_masked1 = x * (1 - mask1)

        im_masked2 = x * (1 - mask2)

        im_masked3 = x * (1 - mask3)

        im_masked4 = x * (1 - mask4)

        y = y1*mask1+y2*mask2+y3*mask3+y4*mask4
        y = (y +x)/2

        return y1,  im_masked1, y2, im_masked2, y3, im_masked3, y4, im_masked4, y
    else:
        loss1, y1, mask1, loss2, y2, mask2 = model(x.float(), mask_ratio)
        y1 = model.unpatchify(y1)
        y1 = torch.einsum('nchw->nhwc', y1).detach().cpu()

        y2 = model.unpatchify(y2)
        y2 = torch.einsum('nchw->nhwc', y2).detach().cpu()


        # visualize the mask
        mask1 = mask1.detach()
        mask1 = mask1.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        mask1 = model.unpatchify(mask1)  # 1 is removing, 0 is keeping
        mask1 = torch.einsum('nchw->nhwc', mask1).detach().cpu()

        mask2 = mask2.detach()
        mask2 = mask2.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        mask2 = model.unpatchify(mask2)  # 1 is removing, 0 is keeping
        mask2 = torch.einsum('nchw->nhwc', mask2).detach().cpu()

        x = torch.einsum('nchw->nhwc', x)

        # masked imagea
        im_masked1 = x * (1 - mask1)

        im_masked2 = x * (1 - mask2)

        Y1 = y1*mask1 + im_masked1

        Y2 = y2*mask2 + im_masked2

        y = (y1+y2)/2

        return Y1, im_masked1,  Y2, im_masked2, y







if __name__ == '__main__':

    advs = attack_img(or_img_dir, label_dir)
    save_adv_img(or_img_dir, advs, adv_dir)
    print('The adversarial example is saved')
    # chkpt_dir = 'checkpoints/mae_visualize_vit_large.pth'
    chkpt_dir = 'checkpoints/mae_visualize_vit_large_ganloss.pth'
    mask_ratio = 0.50
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    data, num = load_adv_img(adv_dir)
    for batch, (img) in enumerate(data):
        torch.manual_seed(2)
        print('MAE with pixel reconstruction:')
        if mask_ratio == 0.25:
            y1, im_masked1, y2, im_masked2, y3, im_masked3, y4, im_masked4, y = run_one_image(img, model_mae, mask_ratio)
        else:
            y1, im_masked1, y2, im_masked2, y = run_one_image(img, model_mae, mask_ratio)

        re_imgs.append(y[0])

    save_reconstruct_img(or_img_dir, re_imgs, save_dir1)
if mask_ratio == 0.25:
    plt.rcParams['figure.figsize'] = [12, 12]
    plt.subplot(2, 3, 1)
    show_image(img[0], "adversarial")

    plt.subplot(2, 3, 2)
    show_image(im_masked1[0], "masked1")

    plt.subplot(2, 3, 3)
    show_image(im_masked2[0], "masked2")

    plt.subplot(2, 3, 4)
    show_image(im_masked3[0], "masked3")

    plt.subplot(2, 3, 5)
    show_image(im_masked4[0], "masked4")
    plt.subplot(2, 3, 6)
    show_image(y[0], "reconstruction ")
    plt.show()
else:
    plt.rcParams['figure.figsize'] = [12, 12]
    plt.subplot(2, 2, 1)
    show_image(img[0], "adversarial")

    plt.subplot(2, 2, 2)
    show_image(y1[0], "masked1")

    plt.subplot(2, 2, 3)
    show_image(y2[0], "masked2")
    plt.subplot(2, 2, 4)
    show_image(y[0], "reconstruction ")
    plt.show()




