from torchvision import transforms
from torch.utils import data
import os
import cv2
import numpy as np
import scipy.io as scio
from PIL import Image

class ImageLoader_for_train_mask2(data.Dataset):
    def __init__(self, data_dir, image_size, transform=None):
        super(ImageLoader_for_train_mask, self).__init__()
        self.data_dir = data_dir
        print(data_dir)
        self.transform = transform
        self.img_paths = os.listdir(data_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        curr_img_path = self.img_paths[index]

        curr_img_path = self.data_dir+curr_img_path

        data_img = Image.open(curr_img_path)
        data_img = self.transform(data_img)
        data_img = np.array(data_img)

        # Sobel 算子进行边缘检测
        sobelx = cv2.Sobel(data_img, cv2.CV_64F, 1, 0, ksize=3)  # 在 x 方向上的 Sobel 算子
        sobely = cv2.Sobel(data_img, cv2.CV_64F, 0, 1, ksize=3)  # 在 y 方向上的 Sobel 算子

        # 计算边缘强度
        edge_intensity = np.sqrt(sobelx ** 2 + sobely ** 2)
        # 归一化处理
        edge_intensity = (edge_intensity - np.min(edge_intensity)) / (np.max(edge_intensity) - np.min(edge_intensity))
        # 反转0和1
        edge_intensity = 1 - edge_intensity

        mask = np.float32(edge_intensity)
        mask = (mask.transpose((2, 0, 1)) + 1) * 3

        m = np.random.rand(30)
        m[m >= 0.5] = 1
        m[m < 0.5] = 0

        Data_img = data_img[:, :, :]

        Data_img = Data_img.transpose((2, 0, 1))
        Data_img = np.float32(Data_img / 255 * 2 - 1)

        return Data_img, m, mask

class ImageLoader_for_train_mask(data.Dataset):
    def __init__(self,data_dir,image_size,transform=None):
        super(ImageLoader_for_train_mask,self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = os.listdir(data_dir)
        self.image_size = image_size
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,index):
        imagesize = self.image_size
        curr_img_path = self.img_paths[index]
        img = cv2.imread(self.data_dir + curr_img_path,1)
        data_img = img[:,:imagesize,:]

        mask = img[:,imagesize:,:]
        mask = np.float32(mask)/255.0
        mask = (mask.transpose((2,0,1))+1)*3

        m = np.random.rand(30)
        m[m>=0.5] = 1
        m[m<0.5] = 0

        Data_img = data_img[:,:,:]

        Data_img = Data_img.transpose((2,0,1))
        Data_img = np.float32(Data_img/255*2-1)
        return  Data_img, m, mask

class ImageLoader_for_test(data.Dataset):
    def __init__(self,data_dir,image_size,w_path,transform=None):
        super(ImageLoader_for_test,self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = os.listdir(data_dir)
        self.image_size = image_size
        W = scio.loadmat(w_path)
        self.w = W['w']
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,index):
        imagesize = self.image_size
        curr_img_path = self.img_paths[index]
        nn = curr_img_path.index('.')
        num = int(curr_img_path[:nn])
        img = cv2.imread(self.data_dir + curr_img_path,1)
        data_img = cv2.resize(img,(imagesize,imagesize))
        m = self.w[num,:]
        Data_img = data_img[:,:,:]

        Data_img = Data_img.transpose((2,0,1))
        Data_img = np.float32(Data_img/255*2-1)
        return  Data_img, m, num

def get_loader(image_dir, image_size=128, 
               batch_size=32, dataset='train_mask', mode='train_mask', num_workers=1, w_path = 'results/WatermarkMatrix/w.mat'):
    """Build and return a data loader."""
    # transform = transforms.Compose([
    #     transforms.Resize(size=(image_size, image_size)),
    # ])
    transform = []

    if dataset in ['train_mask']:
        dataset = ImageLoader_for_train_mask(image_dir,image_size,transform)
    elif dataset in ['test_accuracy','test_embedding']:
        dataset = ImageLoader_for_test(image_dir,image_size,w_path)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=False)
    return data_loader