from model import Encoder_Decoder
from torch.autograd import Variable
import torch
import numpy as np
import os
import cv2
from tqdm.auto import tqdm
from torchvision import transforms

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

model_path = "/projects/pimog2/models/Encoder_Decoder_Model_mask_ScreenShooting_2023-12-26-23-19_70_best_100.pth"

def transform_size(img, image_size):
    transform = transforms.Compose([
         transforms.Resize(size=(image_size, image_size)),
    ])

    return transform(img)

def single_embedding(img_path, model_path):
    net_ED = Encoder_Decoder("distortion").to(device)
    # net_ED = torch.nn.DataParallel(net_ED)

    checkpoint = torch.load(model_path)

    weights_dict = {}
    for k, v in checkpoint.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    net_ED.load_state_dict(weights_dict)
    net_ED = net_ED.to(device)

    net_E = net_ED.Encoder
    net_E = net_E.to(device)
    # net_E = torch.nn.DataParallel(net_E)
    net_E.eval()

    img = cv2.imread(img_path, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.transpose((2, 0, 1))
    img = np.float32(img / 255 * 2 - 1)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = Variable(img)
    img = img.to(device)

    img = transform_size(img, 128)

    m = np.random.rand(30)
    m[m >= 0.5] = 1
    m[m < 0.5] = 0
    print(f"Watermark: {m}")
    m = torch.from_numpy(m)

    m = Variable(m.float())
    m.to(device)

    img_embedding = net_E(img, m)
    img_embedding = (img_embedding[:, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
    img_embedding = img_embedding.squeeze(0)
    img_embedding = np.transpose(img_embedding, (1, 2, 0))

    cv2.imwrite('/projects/pimog2/results/test1.png', img_embedding)

def single_extracting(img_path, model_path):
    net_ED = Encoder_Decoder("distortion").to(device)

    checkpoint = torch.load(model_path)

    weights_dict = {}
    for k, v in checkpoint.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    net_ED.load_state_dict(weights_dict)
    net_ED = net_ED.to(device)

    net_D = net_ED.Decoder
    net_D = net_D.to(device)

    net_D.eval()

    img = cv2.imread(img_path, 1)

    img = img.transpose((2, 0, 1))
    img = np.float32(img / 255 * 2 - 1)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = Variable(img)
    img = img.to(device)
    img = transform_size(img, 128)

    msg = net_D(img)
    msg = msg.detach().cpu().numpy().round().clip(0, 1)
    print(f"Extract: {msg.squeeze(0)}")
    origin = [1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
 1., 1., 0., 1., 0., 1.]
    e = msg.squeeze(0)
    print(origin == e)


def main():
    img_path = "/projects/pimog2/results/s1.jpg"
    # single_embedding(img_path, model_path)
    single_extracting(img_path, model_path)


if __name__ == '__main__':
    main()