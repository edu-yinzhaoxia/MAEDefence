import os

import torch
from torch import nn
from torchattacks.attack import Attack
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

class SINIM(Attack):
	def __init__(self, model, eps=8/255, steps=5, decay=1.0, m=5, bool=1):
		super().__init__("NISIFGSM", model)
		self.eps = eps
		self.decay = decay
		self.steps = steps
		self.m = m
		self.bool = bool
		self._supported_mode = ['default', 'targeted']

	def NIM(self, adv_images,images, labels,alpha,loss,momentum):
		for _ in range(self.steps):
			adv_images.requires_grad = True
			x_nes = adv_images + alpha * self.decay * momentum
			x_nes.requires_grad = True
			outputs = self.model(x_nes)
			cost = loss(outputs, labels)
			g = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
			g = g / torch.norm(g, p=1, dim=(1, 2, 3), keepdim=True)
			grad = g + momentum * self.decay
			momentum = grad

			adv_images = adv_images.detach() + alpha * grad.sign()
			delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
			adv_images = torch.clamp(images + delta, min=0, max=1).detach()

		return adv_images

	def NISIM(self, adv_images, images, labels, alpha, loss, momentum):

		for _ in range(self.steps):
			adv_images.requires_grad = True
			x_nes = adv_images + alpha * self.decay * momentum
			x_nes.requires_grad = True
			g = torch.zeros_like(images).detach().to(self.device)
			for i in range(self.m):
				x_temp = (x_nes / (2**i)).detach()
				x_temp.requires_grad = True
				outputs_temp = self.model(x_temp)
				loss_temp = loss(outputs_temp, labels)
				loss_temp.backward()
				g += x_temp.grad.detach()
			g = g / self.m
			g = g / torch.norm(g, p=1, dim=(1, 2, 3), keepdim=True)
			grad = self.decay * momentum + g
			momentum = grad

			adv_images = adv_images.detach() + alpha * grad.sign()
			delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
			adv_images = torch.clamp(images + delta, min=0, max=1).detach()

		return adv_images



	def forward(self, images, labels):

		images = images.clone().detach().to(self.device)
		labels = labels.clone().detach().to(self.device)
		alpha = self.eps / self.steps
		momentum = torch.zeros_like(images).detach().to(self.device)
		loss = nn.CrossEntropyLoss()
		adv_images = images.clone().detach()
		if self.bool == 1:
			adv_images = self.NIM(labels,images, adv_images, loss, alpha, momentum )
		else:
			adv_images = self.NISIM(labels, images, adv_images, loss, alpha, momentum)

		return adv_images







