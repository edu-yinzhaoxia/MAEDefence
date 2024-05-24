from skimage.restoration import (denoise_wavelet)
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

sigma = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06]  # chose a smaller sigma for small perturbation sizes

def wavelet_denoise(adv_img_dir,denoise_dir):
    filelist = os.listdir(adv_img_dir)
    for j, (filename) in enumerate(filelist):
        imgpath = os.path.join(adv_img_dir, filename)
        img = Image.open(imgpath).resize((224, 224))
        img_rgb = np.array(img)
        im_bayes = denoise_wavelet(img_rgb / 255, multichannel=True, convert2ycbcr=True,
                               method='BayesShrink', mode='soft', sigma=sigma[5])
        image = np.clip(im_bayes, 0, 1)
        imgpath1 = os.path.join(denoise_dir, filename)
        plt.imsave(imgpath1, image)

