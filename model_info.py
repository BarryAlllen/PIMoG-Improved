from model256 import Discriminator
from model256 import Encoder_Decoder
from model256 import U_Net_Encoder_Diffusion
from torchinfo import summary
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

encoder_input_size = [(1, 3, 128, 128), (1, 15)]
discriminator_input_size = [1, 3, 1, 2]

def encoder_info(input_size):
    model = U_Net_Encoder_Diffusion().to(device)
    print(model)
    summary(model, input_size=input_size)

def disciminator_info(input_size):
    model = Discriminator(64).to(device)
    print(model)
    summary(model, input_size=input_size)

def main():
    # encoder_info(encoder_input_size)
    disciminator_info(discriminator_input_size)

if __name__ == '__main__':
    main()


