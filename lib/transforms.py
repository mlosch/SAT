import torch
import torch.nn as nn
import numpy as np

class PatchifyImage(nn.Module):
	def __init__(self, num_patches):
		super(PatchifyImage, self).__init__()
		self.num_patches = num_patches

	def forward(self, img):
		grid_n = int(np.sqrt(self.num_patches))
		assert img.shape[-1] % grid_n == 0
		patch_size = img.shape[-1] // grid_n
		img = torch.nn.functional.unfold(img.unsqueeze(0), patch_size, stride=patch_size).squeeze(0)
		img=img.view(3,img.shape[0]//3,img.shape[1])
		img=img.permute(2,0,1).contiguous()
		img=img.view(img.shape[0], img.shape[1], patch_size, patch_size)
		return img