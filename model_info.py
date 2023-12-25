from model128 import Discriminator
from model128 import Decoder
# from model128 import U_Net_Encoder_Diffusion
from model256 import U_Net_Encoder_Diffusion
from torchinfo import summary
from Noise_Layer import Identity, ScreenShooting
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

encoder_input_size = [(1, 3, 128, 128), (1, 15)]
encoder_input_size2 = [(1, 3, 256, 256), (1, 30)]
decoder_input_size = [1, 3, 256, 256]
discriminator_input_size = [1, 3, 1, 2]
noise_input_size = [3, 1, 1, 2]

def encoder_info(input_size):
    model = U_Net_Encoder_Diffusion().to(device)
    print(model)
    summary(model, input_size=input_size)

def decoder_info(input_size):
    model = Decoder().to(device)
    print(model)
    summary(model, input_size=input_size)

def disciminator_info(input_size):
    model = Discriminator(64).to(device)
    print(model)
    summary(model, input_size=input_size)

def noise_info(input_size):
    model = ScreenShooting().to(device)
    print(model)
    summary(model, input_size=input_size)

def main():
    encoder_info(encoder_input_size2)
    # decoder_info(decoder_input_size)
    # disciminator_info(discriminator_input_size)
    # noise_info(noise_input_size)

if __name__ == '__main__':
    main()


