import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import random
from  io import BytesIO
from torchvision import transforms

def load_image_with_cache(path, cache=None, lock=None):
	if cache is not None:
		if not path in cache:        #if not cache.has_key(path):
			with open(path, 'rb') as f:
				cache[path] = f.read()
		return Image.open(BytesIO(cache[path]))
	return Image.open(path)

class Data(data.Dataset):
	def __init__(self, root, lst, image_size, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892]),
		crop_size=None, rgb=True, scale=None):
		self.mean_bgr = mean_bgr
		self.root = root
		self.lst = lst
		self.yita = yita
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}
		self.image_size=image_size
		self.transform_size = transforms.Compose([
			transforms.Resize(size=(self.image_size, self.image_size)),
		])

		lst_dir = os.path.join(self.root, self.lst)
		# self.files = np.loadtxt(lst_dir, dtype=str)
		with open(lst_dir, 'r') as f:
			self.files = f.readlines()
			self.files = [line.strip().split(' ') for line in self.files]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		img_file = self.root + data_file[0]

		if not os.path.exists(img_file):
			img_file = img_file.replace('jpg', 'png')

		img = load_image_with_cache(img_file, self.cache)
		# load gt image
		gt_file = self.root + data_file[0]

		gt = load_image_with_cache(gt_file, self.cache)
		if gt.mode == '1':
			gt  = gt.convert('L')

		img = self.transform_size(img)
		gt = self.transform_size(gt)

		return self.transform(img, gt), np.array(img, dtype=np.float32)

	def transform(self, img, gt):
		gt = np.array(gt, dtype=np.float32)
		if len(gt.shape) == 3:
			gt = gt[:, :, 0]
		gt /= 255.
		gt[gt >= self.yita] = 1
		gt = torch.from_numpy(np.array([gt])).float()
		img = np.array(img, dtype=np.float32)
		if self.rgb:
			img = img[:, :, ::-1] # RGB->BGR
		img -= self.mean_bgr
		data = []
		if self.scale is not None:
			for scl in self.scale:
				img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
				data.append(torch.from_numpy(img_scale.transpose((2,0,1))).float())
			return data, gt
		img = img.transpose((2, 0, 1))
		img = torch.from_numpy(img.copy()).float()
		if self.crop_size:
			_, h, w = gt.size()
			assert(self.crop_size < h and self.crop_size < w)
			i = random.randint(0, h - self.crop_size)
			j = random.randint(0, w - self.crop_size)
			img = img[:, i:i+self.crop_size, j:j+self.crop_size]
			gt = gt[:, i:i+self.crop_size, j:j+self.crop_size]
		return img, gt


