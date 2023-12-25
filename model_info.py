from model import Discriminator
from model import Encoder_Decoder
from model import U_Net_Encoder_Diffusion
from torchinfo import summary

encoder_input_size = [(1, 3, 128, 128), (1, 15)]

def encoder_info(input_size):
    encoder = U_Net_Encoder_Diffusion()
    print(encoder)
    summary(encoder, input_size=encoder_input_size)

def main():
    encoder_info(encoder_input_size)

if __name__ == '__main__':
    main()


