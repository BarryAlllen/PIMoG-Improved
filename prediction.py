from model import Encoder_Decoder
from torch.autograd import Variable
import torch
import numpy as np
import argparse
import cv2
from torchvision import transforms
from torch.utils import data
from data_loader import ImageLoade_for_prediction
from tqdm import tqdm
from Noise_Layer import ScreenShooting

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def get_model(model_path):
    net_ED = Encoder_Decoder("distortion").to(device)

    checkpoint = torch.load(model_path)

    weights_dict = {}
    for k, v in checkpoint.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    net_ED.load_state_dict(weights_dict)
    net_ED = net_ED.to(device)
    return net_ED

def transform_size(img, image_size):
    transform = transforms.Compose([
         transforms.Resize(size=(image_size, image_size)),
    ])
    return transform(img)

def extracting(img_path, model_path, img_size=128, is_noise=False):
    net_ED = get_model(model_path)

    net_D = net_ED.Decoder
    net_D = net_D.to(device)

    net_D.eval()

    dataset = ImageLoade_for_prediction(img_path)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  num_workers=1
                                  )
    watermark = {}
    with torch.inference_mode():
        for i, (img, img_name) in tqdm(enumerate(data_loader), desc="Decoding", total=len(data_loader)):
            img_name = img_name[0]
            img_name = "["+img_name+"]"
            img = transform_size(img, img_size)
            if is_noise:
                img = noise(img)
            img = img.float()
            img = img.to(device)
            msg = net_D(img)
            msg = msg.detach().cpu().numpy().round().clip(0, 1)
            msg = msg.squeeze(0)
            msg = ''.join(str(int(x)) for x in msg)
            watermark[img_name] = msg
        for k,v in watermark.items():
            print(f"{k}: {v}")


def embedding(img_path, output, model_path, img_size=128):

    net_ED = get_model(model_path)
    net_E = net_ED.Encoder
    net_E = net_E.to(device)
    net_E.eval()

    dataset = ImageLoade_for_prediction(img_path)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  num_workers=1
                                  )

    with torch.inference_mode():
        for i, (img, img_name) in tqdm(enumerate(data_loader), desc="Encoding", total=len(data_loader)):
            img = transform_size(img, img_size)
            img = img.to(device)

            m = get_msg(False)

            img_embedding = net_E(img, m)
            img_embedding = (img_embedding[:, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
            img_embedding = img_embedding.squeeze(0)
            img_embedding = np.transpose(img_embedding, (1, 2, 0))
            img_name = img_name[0].split(".")[0]
            img_name = img_name + ".png"
            cv2.imwrite(output+img_name, img_embedding)

def get_msg(is_random=False):
    if is_random:
        m = np.random.rand(30)
        m[m >= 0.5] = 1
        m[m < 0.5] = 0
    else:
        m = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])

    m = torch.from_numpy(m)
    m = Variable(m.float())
    m.to(device)
    return m

def noise(img):
    Noiser = ScreenShooting()
    noise_img = Noiser(img)
    return noise_img

def main(config):
    mode = config.mode
    model = config.model
    if mode == "encode":
        print("== Encoding Mode ==")
        img_path = config.encode_img_path
        output = config.encode_result_path
        embedding(img_path, output, model,128)
    elif mode == "decode":
        print("== Decoding Mode ==")
        img = config.decode_img_path
        extracting(img, model,128,True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="decode",choices=['encode','decode'], help='mode')
    parser.add_argument('--model', type=str, default="models/ScreenShooting/model.pth",choices=['encode','decode'], help='mode')
    parser.add_argument('--decode_img_path', type=str, default="results/res/", help='img path')
    parser.add_argument('--encode_img_path', type=str, default="images/", help='img path')
    parser.add_argument('--encode_result_path', type=str, default="results/res/", help='img encode result path')
    config = parser.parse_args()
    main(config)