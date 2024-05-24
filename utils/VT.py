import os

import torch
from torch import nn
from torchattacks.attack import Attack

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

class VMIM(Attack):
    def __init__(self, model, number=20, beta=1.5, eps=8 / 255, steps=5, decay=1.0):
        super().__init__("VMIFGSM", model)
        self.number = number         # 方差调整选取周边点的数量
        self.beta = beta
        self.eps = eps
        self.steps = steps           # 对抗攻击的迭代步数
        self.decay = decay
        self._supported_mode = ['default', 'targeted']

    def get_sumgrad(self, labels, adv_images, gama, loss, new_grad):
        min_value = -gama
        max_value = gama
        sum_grad =torch.zeros_like(new_grad).detach().to(self.device)
        for i in range(self.number):
            images_neighbor = adv_images + torch.empty_like(adv_images).uniform_(min_value, max_value)
            images_neighbor = images_neighbor.clone().detach().to(self.device)
            images_neighbor.requires_grad = True
            outputs = self.model(images_neighbor)
            cost = loss(outputs,labels)
            grad = torch.autograd.grad(cost, images_neighbor,
                                       retain_graph=False, create_graph=False)[0]
            sum_grad += grad
        return sum_grad

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        alpha = self.eps / self.steps
        gama = self.beta*alpha

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        variance = torch.zeros_like(adv_images).detach().to(self.device)
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            new_grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]


            sum_grad = self.get_sumgrad(labels, adv_images, gama, loss, new_grad)

            current_grad = new_grad + variance
            current_grad = current_grad / torch.mean(torch.abs(current_grad), dim=(1, 2, 3), keepdim=True)
            variance = sum_grad / (1.*self.number) - new_grad
            grad = current_grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images