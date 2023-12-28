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

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

model_path = "/projects/pimog2/models/Encoder_Decoder_Model_mask_ScreenShooting_2023-12-26-23-19_70_best_100.pth"

def transform_size(img, image_size):
    transform = transforms.Compose([
         transforms.Resize(size=(image_size, image_size)),
    ])
    return transform(img)

def embedding2(img_path, output, model_path):
    net_ED = Encoder_Decoder("distortion").to(device)

    checkpoint = torch.load(model_path)

    weights_dict = {}
    for k, v in checkpoint.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    net_ED.load_state_dict(weights_dict)
    net_ED = net_ED.to(device)

    net_E = net_ED.Encoder
    net_E = net_E.to(device)
    net_E.eval()

    img = cv2.imread(img_path, 1)
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
    # print(f"Watermark: {m}")
    m = torch.from_numpy(m)

    m = Variable(m.float())
    m.to(device)

    img_embedding = net_E(img, m)
    img_embedding = (img_embedding[:, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
    img_embedding = img_embedding.squeeze(0)
    img_embedding = np.transpose(img_embedding, (1, 2, 0))

    cv2.imwrite(output, img_embedding)
    print("Complete encode!")

def extracting2(img_path, model_path):
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
    msg = msg.squeeze(0)
    msg = ''.join(str(int(x)) for x in msg)
    print(f"Watermark: {msg}")
    print("Complete decode!")

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

            m = np.random.rand(30)
            m[m >= 0.5] = 1
            m[m < 0.5] = 0
            # print(f"Watermark: {m}")
            m = torch.from_numpy(m)

            m = Variable(m.float())
            m.to(device)

            img_embedding = net_E(img, m)
            img_embedding = (img_embedding[:, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
            img_embedding = img_embedding.squeeze(0)
            img_embedding = np.transpose(img_embedding, (1, 2, 0))
            img_name = img_name[0].split(".")[0]
            img_name = img_name + ".png"
            cv2.imwrite(output+img_name, img_embedding)


def main(config):
    mode = config.mode
    model = config.model
    if mode == "encode":
        print("Encoding...")
        img_path = config.encode_img_path
        output = config.encode_result_path
        embedding(img_path, output, model,128)
    elif mode == "decode":
        print("Decoding...")
        img = config.decodeimg
        # extracting(img, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="encode",choices=['encode','decode'], help='mode')
    parser.add_argument('--model', type=str, default="models/ScreenShooting/model.pth",choices=['encode','decode'], help='mode')
    parser.add_argument('--encodeimg', type=str, default="results/Image/images/3967.jpg", help='img path')
    parser.add_argument('--encodeout', type=str, default="results/Image/3967.png", help='result path')
    parser.add_argument('--decodeimg', type=str, default="results/Image/images/3967.png", help='img path')
    parser.add_argument('--encode_img_path', type=str, default="images/", help='img path')
    parser.add_argument('--encode_result_path', type=str, default="results/Image/", help='img encode result path')
    config = parser.parse_args()
    main(config)