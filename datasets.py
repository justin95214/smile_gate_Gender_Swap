import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms



def to_rgb(image):
	rgb_image = Image.new("RGB", image.size)
	rgb_image.paste(image)

	return rgb_image


class Dataset():
	def __init__(self, root, transforms_ =None, unaligned=False, mode="train"):
		self.transform = transforms.Compose(transforms_)
		self.unaligned = unaligned
		#path = os.path.join(root, "%s/man" % mode)
		#print(path)

		self.files_man = sorted(glob.glob(os.path.join(root, "%s/male" % mode) + "/*.*"))
		self.files_woman = sorted(glob.glob(os.path.join(root, "%s/female" % mode) + "/*.*"))


	def __getitem__(self, index):
		image_man =Image.open(self.files_man[index % len(self.files_man)])

		if self.unaligned:
			image_woman = Image.open(self.files_woman[random.randint(0, len(self.files_woman) - 1)])
		else:
			image_woman = Image.open(self.files_woman[index % len(self.files_woman)])


		if image_man.mode !="RGB":
			image_man = to_rgb(image_man)
		if image_woman !="RGB":
			image_woman = to_rgb(image_woman)

		item_man = self.transform(image_man)
		item_woman = self.transform(image_woman)
		
		return {"A": item_man, "B": item_woman}


	def __len__(self):
		return max(len(self.files_man), len(self.files_woman))

