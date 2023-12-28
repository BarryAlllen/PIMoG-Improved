import numpy as np
import torch
from torch.autograd import Variable
import os
import cv2
import bdcn
from dataset import Data
import argparse
from tqdm import tqdm
from generate_filename_txt import write_filenames_to_txt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sigmoid(x):
    return 1. / (1 + np.exp(np.array(-1. * x)))

def get_edge_mask(model, args):
    img_root = args.imgroot
    img_lst = args.imglst
    write_filenames_to_txt(img_root, img_lst)
    img_name_lst = os.path.join(img_root, img_lst)
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    test_img = Data(img_root, img_lst, args.imgsize, mean_bgr=mean_bgr)
    testloader = torch.utils.data.DataLoader(
        test_img, batch_size=1, shuffle=False, num_workers=8)  # num_workers=8
    nm = np.loadtxt(img_name_lst, dtype=str)
    nm = np.array([row[0] for row in nm])

    save_dir = args.results
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model.to(device)
    model.eval()
    iter_per_epoch = len(testloader)
    with torch.inference_mode():
        for i, ((data, _), origin_img) in tqdm(enumerate(testloader), total=iter_per_epoch, desc="Process image"):
            data = data.to(device)
            data = Variable(data)
            out = model(data)
            fuse = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
            fuse = 255 - fuse * 255
            fuse = torch.from_numpy(fuse)
            I1 = origin_img
            I1 = I1.squeeze(0)
            I1 = I1.detach().to('cpu').numpy()
            I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
            I2 = fuse.unsqueeze(2)
            I_res = np.zeros((I1.shape[0],I1.shape[1]*2,I1.shape[2]))
            I_res[:, :I1.shape[1],:] = I1
            I_res[:, I1.shape[1]:, :] = I2
            try:
                cv2.imwrite(os.path.join(save_dir, '%s' % nm[i]), I_res)
            except Exception as e:
                print("not write", i)

def main():
    import time
    print(time.localtime())
    args = parse_args()
    model = bdcn.BDCN()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    get_edge_mask(model, args)

def parse_args():
    parser = argparse.ArgumentParser('BDCN')
    parser.add_argument('--imgroot', type=str, default='/data/datasets/coco/train2017_8/', help='The dataset\'s root')
    parser.add_argument('--imglst', type=str, default='coco_datasets.txt', help='The dataset\'s list')
    parser.add_argument( '--model', type=str, default='/data/model/bdcn/bdcn_pretrained_on_bsds500.pth', help='the model to test')
    parser.add_argument('--results', type=str, default='/data/datasets/coco/train2017_test', help='the dir to store result')
    parser.add_argument('--imgsize', type=int, default=128, help='transform image size')
    return parser.parse_args()


if __name__ == '__main__':
    main()
