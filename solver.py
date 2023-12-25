from model import Discriminator
from model import Encoder_Decoder
from torch.autograd import Variable
import torch.optim as optim
import torch
import numpy as np
import os
import time
import torch.nn as nn
import cv2
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from timeit import default_timer as timer
from datetime import datetime


class Solver(object):
    """Solver for training and testing PIMoG."""

    def __init__(self, data_loader, data_loader_test, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        # Model configurations.
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2
        self.lambda3 = config.lambda3
        self.num_epoch = config.num_epoch
        self.distortion = config.distortion

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
        self.model_name = config.model_name
        self.result_dir = config.result_dir
        self.embedding_epoch = config.embedding_epoch

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step

        # Build the model.
        self.build_model()

    def build_model(self):
        if self.dataset in ['train_mask']:
            self.net = Encoder_Decoder(self.distortion)
            self.net_Discriminator = Discriminator(self.num_channels)
            self.net_Discriminator.to(self.device)
            self.optimizer_Discriminator = optim.Adam(self.net_Discriminator.parameters())
            self.net_optimizer = torch.optim.Adam(self.net.parameters())
            self.print_network(self.net, self.dataset)
            self.net.to(self.device)
            self.net = torch.nn.DataParallel(self.net)
            if self.embedding_epoch != 0:
                self.net.load_state_dict(torch.load(
                    self.model_save_dir + '/' + self.distortion + '/' + self.model_name + '_mask_' + str(
                        self.embedding_epoch) + '.pth'))
        elif self.dataset in ['test_embedding']:
            self.net_ED = Encoder_Decoder(self.distortion)
            self.net_ED = self.net_ED.to(self.device)
            self.net_E = self.net_ED.Encoder
            self.net_ED = torch.nn.DataParallel(self.net_ED)
            self.net_ED.load_state_dict(torch.load(
                self.model_save_dir + '/' + self.distortion + '/' + self.model_name + '_mask_' + str(
                    self.embedding_epoch) + '.pth'))
        elif self.dataset in ['test_accuracy']:
            self.net = Encoder_Decoder(self.distortion)
            self.print_network(self.net, self.dataset)
            self.net.to(self.device)
            self.net_D = self.net.Decoder
            self.net = torch.nn.DataParallel(self.net)
            self.net.load_state_dict(torch.load(
                self.model_save_dir + '/' + self.distortion + '/' + self.model_name + '_mask_' + str(
                    self.embedding_epoch) + '.pth'))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

    def test_embedding(self):
        # Set data loader.
        data_loader = self.data_loader
        data_loader_test = self.data_loader_test
        criterion_MSE = nn.MSELoss()
        self.net_ED.eval()
        for i, (data, m, num) in enumerate(data_loader):
            inputs, m = Variable(data), Variable(m.float())
            inputs, m = inputs.to(self.device), m.to(self.device)
            inputs.requires_grad = True
            num = num.to('cpu').numpy()
            Encoded_image, Noised_image, Decoded_message = self.net_ED(inputs, m)
            loss_de = criterion_MSE(Decoded_message, m)
            loss_de.backward()
            inputgrad = inputs.grad.data
            mask = torch.zeros(inputgrad.shape).to(self.device)
            for ii in range(inputgrad.shape[0]):
                a = inputgrad[ii, :, :, :]
                a = (1 - (a - a.min()) / (a.max() - a.min())) + 1
                mask[ii, :, :, :] = a

            for j in range(Encoded_image.shape[0]):
                I1 = (inputs[j, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
                I1 = np.transpose(I1, (1, 2, 0))
                I2 = (Encoded_image[j, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
                I2 = np.transpose(I2, (1, 2, 0))
                I_no = (Noised_image[j, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
                I_no = np.transpose(I_no, (1, 2, 0))
                I_mask = (mask[j, :, :, :].detach().to('cpu').numpy() - 1) * 255
                I_mask = np.transpose(I_mask, (1, 2, 0))
                I_res = (I2 - I1) * 5
                I5 = np.zeros((I1.shape[0], I1.shape[1] * 5, I1.shape[2]))
                I5[:, :I1.shape[1], :] = I1
                I5[:, I1.shape[1]:I1.shape[1] * 2, :] = I2
                I5[:, I1.shape[1] * 2:I1.shape[1] * 3, :] = I_no
                I5[:, I1.shape[1] * 3:I1.shape[1] * 4, :] = I_mask
                I5[:, I1.shape[1] * 4:I1.shape[1] * 5, :] = I_res
                index = num[j]
                if not os.path.exists(self.result_dir + '/Image_test_' + self.distortion + '/images_embed_' + str(
                        self.embedding_epoch) + '/'):
                    os.makedirs(self.result_dir + '/Image_test_' + self.distortion + '/images_embed_' + str(
                        self.embedding_epoch) + '/')
                cv2.imwrite(self.result_dir + '/Image_test_' + self.distortion + '/images_embed_' + str(
                    self.embedding_epoch) + '/' + str(index) + '.png', I5)
        print('Embed finished!')

    def train_mask(self):
        current_time = datetime.now()
        current_time = current_time.strftime("%Y-%m-%d-%H-%M")

        # Set data loader.
        data_loader = self.data_loader
        data_loader_test = self.data_loader_test
        criterion = nn.BCEWithLogitsLoss()
        criterion_MSE = nn.MSELoss()
        start_epoch = self.embedding_epoch

        # Start training.
        print('Start training...')
        start_time = time.time()
        txtfile = open(self.log_dir + '/' + self.dataset + '_' + self.distortion + '_' + current_time + '.txt', 'w',
                       encoding="utf-8")
        best_acc = 0.3

        tensorboard_logname = current_time
        if not os.path.exists(self.log_dir + '/' + tensorboard_logname + '/'):
            os.makedirs(self.log_dir + '/' + tensorboard_logname + '/')
        tensor_board = SummaryWriter("logs/" + tensorboard_logname)
        batch_count = 1
        show_per = 1
        total_corrcet = 0

        for epoch in tqdm(range(start_epoch, self.num_epoch), desc="Total"):
            running_loss = 0.0
            message_losses = 0.0
            denoise_losses = 0.0
            g_losses = 0.0
            d_losses = 0.0

            total_batch_time = 0

            for i, (data, m, v_mask) in enumerate(data_loader):
                start_batch_time = timer()

                inputs, m, v_mask = Variable(data), Variable(m.float()), Variable(v_mask)
                inputs, m, v_mask = inputs.to(self.device), m.to(self.device), v_mask.to(self.device)
                inputs.requires_grad = True
                Encoded_image, Noised_image, Decoded_message = self.net(inputs, m)
                loss_de = criterion_MSE(Decoded_message, m)
                inputgrad = torch.autograd.grad(loss_de, inputs, create_graph=True)[0]
                mask = torch.zeros(inputgrad.shape).to(self.device)
                for ii in range(inputgrad.shape[0]):
                    a = inputgrad[ii, :, :, :]
                    a = (1 - (a - a.min()) / (a.max() - a.min())) + 1
                    mask[ii, :, :, :] = a.detach()
                d_label_host = torch.full((inputs.shape[0], 1), 1, dtype=torch.float, device=self.device)
                d_label_decoded = torch.full((inputs.shape[0], 1), 0, dtype=torch.float, device=self.device)
                g_label_decoded = torch.full((inputs.shape[0], 1), 1, dtype=torch.float, device=self.device)

                # train the discriminator
                self.optimizer_Discriminator.zero_grad()
                d_image = self.net_Discriminator(inputs.detach())
                d_loss_host = criterion(d_image, d_label_host)
                d_loss_host.backward()

                d_decoded = self.net_Discriminator(Encoded_image.detach())
                d_loss = criterion(d_decoded, d_label_decoded)
                d_loss.backward()

                self.optimizer_Discriminator.step()

                # train the Encoder_Decoder
                g_decoded = self.net_Discriminator(Encoded_image)
                g_loss = criterion(g_decoded, g_label_decoded)

                loss_message = criterion_MSE(Decoded_message, m)
                loss_denoise = criterion_MSE(Encoded_image * mask.float(), inputs * mask.float()) * 0.5 + criterion_MSE(
                    Encoded_image * v_mask.float(), inputs * v_mask.float()) * 2
                loss = loss_message * self.lambda1 + loss_denoise * self.lambda2 + g_loss * self.lambda3
                self.net_optimizer.zero_grad()
                loss.backward()

                self.net_optimizer.step()

                running_loss += loss.item()
                message_losses += loss_message.item()
                denoise_losses += loss_denoise.item()
                g_losses += g_loss.item()
                d_losses += d_loss.item()

                end_batch_time = timer()
                total_batch_time += (end_batch_time - start_batch_time)

                if (i + 1) % show_per == 0:
                    running_loss /= show_per
                    message_losses /= show_per
                    denoise_losses /= show_per
                    g_losses /= show_per
                    d_losses /= show_per

                    print('[epoch:%d, batch:%d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss))
                    print('[epoch:%d, batch:%d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss), file=txtfile)
                    print('message loss:%.3f, denoise loss:%.3f, gan_loss:%.3f, d_loss:%.3f (%.2f s)' % (
                        message_losses, denoise_losses, g_losses, d_losses, total_batch_time))
                    print('message loss:%.3f, denoise loss:%.3f, gan_loss:%.3f, d_loss:%.3f (%.2f s)' % (
                        message_losses, denoise_losses, g_losses, d_losses, total_batch_time), file=txtfile)
                    total_batch_time = 0

                    tensor_board.add_scalar('1 running loss', running_loss, batch_count)
                    tensor_board.add_scalar('2 message loss', message_losses, batch_count)
                    tensor_board.add_scalar('3 denoise loss', denoise_losses, batch_count)
                    tensor_board.add_scalar('4 d loss', g_losses, batch_count)
                    tensor_board.add_scalar('5 g loss', d_losses, batch_count)

                    running_loss = 0.0
                    message_losses = 0.0
                    denoise_losses = 0.0
                    g_losses = 0.0
                    d_losses = 0.0
                batch_count += 1

            self.net.eval()
            t = np.random.randint(inputs.shape[0])
            I1 = (inputs[t, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
            I1 = np.transpose(I1, (1, 2, 0))
            I2 = (Encoded_image[t, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
            I2 = np.transpose(I2, (1, 2, 0))
            I_no = (Noised_image[t, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
            I_no = np.transpose(I_no, (1, 2, 0))
            I_mask = (mask[t, :, :, :].detach().to('cpu').numpy() - 1) * 255
            I_mask = np.transpose(I_mask, (1, 2, 0))
            I_vmask = (v_mask[t, :, :, :].detach().to('cpu').numpy() - 1) * 50
            I_vmask = np.transpose(I_vmask, (1, 2, 0))
            I_res = (I2 - I1) * 5
            I5 = np.zeros((I1.shape[0], I1.shape[1] * 6, I1.shape[2]))
            I5[:, :I1.shape[1], :] = I1
            I5[:, I1.shape[1]:I1.shape[1] * 2, :] = I2
            I5[:, I1.shape[1] * 2:I1.shape[1] * 3, :] = I_no
            I5[:, I1.shape[1] * 3:I1.shape[1] * 4, :] = I_mask
            I5[:, I1.shape[1] * 4:I1.shape[1] * 5, :] = I_vmask
            I5[:, I1.shape[1] * 5:I1.shape[1] * 6, :] = I_res
            if not os.path.exists(self.result_dir + '/Image/images/' + current_time + '/'):
                os.makedirs(self.result_dir + '/Image/images/' + current_time + '/')
            cv2.imwrite(self.result_dir + '/Image/images/' + current_time + '/' + str(epoch) + '.png', I5)
            imgI5 = cv2.imread(self.result_dir + '/Image/images/' + current_time + '/' + str(epoch) + '.png',
                               cv2.IMREAD_COLOR)
            I5 = cv2.cvtColor(imgI5, cv2.COLOR_BGR2RGB)
            tensor_board.add_image('0 training image', I5, epoch + 1, dataformats="HWC")
            print('validation...')
            print('validation...', file=txtfile)

            correct = 0
            total = 0

            with torch.inference_mode():
                for i, (data, m, v_mask) in enumerate(data_loader_test):
                    inputs, m, v_mask = Variable(data), Variable(m.float()), Variable(v_mask)
                    inputs, m, v_mask = inputs.to(self.device), m.to(self.device), v_mask.to(self.device)
                    Encoded_image, Noised_image, Decoded_message = self.net(inputs, m)
                    decoded_rounded = Decoded_message.detach().cpu().numpy().round().clip(0, 1)
                    correct += np.sum(np.abs(decoded_rounded - m.detach().cpu().numpy()))
                    total += inputs.shape[0] * m.shape[1]

            correct = (1 - correct / total) * 100
            total_corrcet += correct

            print("[epoch:%d] Correct Rate:%.3f" % (epoch + 1, correct) + '%')
            print("[epoch:%d] Correct Rate:%.3f" % (epoch + 1, correct) + '%', file=txtfile)
            tensor_board.add_scalar('0 Correct Rate', correct, epoch + 1)

            if not os.path.exists(self.model_save_dir + '/' + self.distortion + '/' + current_time + '/'):
                os.makedirs(self.model_save_dir + '/' + self.distortion + '/' + current_time + '/')
            PATH_Encoder_Decoder = self.model_save_dir + '/' + self.distortion + '/' + current_time + '/' + self.model_name + '_mask_' + str(
                epoch) + '.pth'

            if epoch % (self.model_save_step) == (self.model_save_step - 1):
                torch.save(self.net.state_dict(), PATH_Encoder_Decoder)

            if 1 - correct / total >= best_acc:
                best_acc = 1 - correct / total
                PATH_Encoder_Decoder_best = self.model_save_dir + '/' + self.model_name + '_mask_' + str(
                    self.distortion) + '_' + current_time + '_best.pth'
                torch.save(self.net.state_dict(), PATH_Encoder_Decoder_best)
            self.net.train()
        print(f"Complete training!!! (Avg acc: {(total_corrcet / len(data_loader)):.2f}%)")

    def test_accuracy(self):
        # Set data loader.
        data_loader_test = self.data_loader_test

        correct = 0
        total = 0
        for i, (data, m, num) in enumerate(data_loader_test):
            inputs, m = Variable(data), Variable(m.float())
            inputs, m = inputs.to(self.device), m.to(self.device)
            self.net_D.eval()
            Decoded_message = self.net_D(inputs)
            decoded_rounded = Decoded_message.detach().cpu().numpy().round().clip(0, 1)
            print(decoded_rounded)
            correct += np.sum(np.abs(decoded_rounded - m.detach().cpu().numpy()))
            total += inputs.shape[0] * m.shape[1]

        print('Accuracy of ' + self.distortion + ' image: %.3f' % ((1 - correct / total) * 100) + '%')
