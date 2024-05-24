import os

from PIL import Image
import torch
import torchvision.transforms as transforms

from utils.load_img import load_adv_img

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])
imagenet_std = torch.Tensor([0.229, 0.224, 0.225])

denoise_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\Re_image'

# filelist = os.listdir(denoise_dir)
def jpeg(dir, save_dir):
    data, num = load_adv_img(dir)
    for batch, (img) in enumerate(data):
        img = img.squeeze()
        img = torch.Tensor.permute(img, (2, 0, 1))
        image = img.clone()
        for i in range(3):
            image[i] = image[i] * imagenet_std[i] + imagenet_mean[i]

        image = torch.clamp(image, min=0, max=1)
        pil_image = transforms.ToPILImage()(image)
        quality = 90  # 压缩质量，范围为0到100
        filename = filelist[batch]
        pil_image.save(os.path.join(save_dir, filename),  format='JPEG', quality=quality)