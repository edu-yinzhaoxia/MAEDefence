import numpy as np
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

from attacks.pre_attack import BIM, FGSM, MIM, PGD, CW, VMI, DIM, TIM, SINIFGSM, VNI
from utils.load_img import load_imagenet_val
from torchvision.models.feature_extraction import create_feature_extractor
from utils.auxiliary_utils import *

or_img_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\val_rs'
adv_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\adversarial_img'
label_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\val_rs.xls'
re_dir = 'C:\\Users\\123\\Desktop\\MAE_defend\datasets\\re_outputs'




normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
device = torch.device("cuda")
classifier = nn.Sequential(
    normalize,
    models.resnet50(pretrained=True))
RENET50 = models.resnet50(pretrained=True)
RENET50 = RENET50.to(device)



def attack_img():
    seed = 18
    if seed != -1:
        print('set seed : ', seed)
        setup_seed(seed)
    data, num_images = load_imagenet_val(or_img_dir, label_dir)
    advs = []

    pbar = tqdm(total=1000 / 50)
    for batch, (input, target) in enumerate(data):
        input = input.to(device)
        target = target.to(device)
        adv = MIM(classifier,  input, target)
    pbar.update(1)


    # data1, num_images = load_imagenet_val(or_img_dir, label_dir)
    # for batch, (re_input, target) in enumerate(data1):
    #     re_input = re_input.to(device)
    #     target = target.to(device)
    #     re_input = re_input.cuda()
    #     target = target.cuda()
    return input, adv












if __name__ == '__main__':
    inputs, advs = attack_img()
    advs = advs.cpu()
    adv = advs[8].to(device).unsqueeze(0)
    input = inputs[8].unsqueeze(0)
    model_trunc = create_feature_extractor(RENET50, return_nodes={'layer4.2.relu_2': 'sematic_feature'})

    or_logit = model_trunc(input)['sematic_feature'].squeeze().detach().cpu()
    adv_logit = model_trunc(adv)['sematic_feature'].squeeze().detach().cpu()


    # 定义两组特征矩阵
    adversarial_features = torch.flatten(adv_logit, start_dim=1).numpy()
    clean_features = torch.flatten(or_logit, start_dim=1).numpy()

    np.random.seed(42)
    n_features = adversarial_features.shape[0]
    rand_features_indices = np.random.choice(n_features, size=500, replace=False)

    adversarial_features = adversarial_features[rand_features_indices, :]
    clean_features = clean_features[rand_features_indices, :]

    # 将两组特征矩阵合并，并进行标准化处理
    features = np.vstack((adversarial_features, clean_features))
    features_standardized = (features - features.mean(axis=0)) / features.std(axis=0)

    # 进行主成分分析，得到3个主成分
    pca = PCA(n_components=3)
    pca.fit(features_standardized)
    transformed_features = pca.transform(features_standardized)

    # 绘制3D散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(transformed_features[0:100, 0], transformed_features[0:100, 1], transformed_features[0:100, 2], c='b',
               label='Reconstruction')
    ax.scatter(transformed_features[100:, 0], transformed_features[100:, 1], transformed_features[100:, 2], c='g',
               label='Clean')

    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    ax.legend()
    plt.show()


# import torchvision
# import torch
# from torchvision.models.feature_extraction import get_graph_node_names
#
# model = torchvision.models.resnet50(pretrained=True)
# nodes, _ = get_graph_node_names(model)
# nodes
# print(nodes)

