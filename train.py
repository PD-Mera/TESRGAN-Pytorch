import os
from math import log10
from statistics import mode
from numpy import real_if_close

import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import argparse

from data import TrainDataset, display_transform, Opt
from model import Discriminator_WGAN, Generator, Discriminator, compute_gradient_penalty
from loss import GeneratorLoss, GeneratorWLoss
from ssim import ssim

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_epochs', default=2,
                    type=int, help='training epoch')
    ap.add_argument('--data_path', default=None,
                    type=str, help='training data path')

    opt_parser = ap.parse_args()

    opt = Opt()
    LR_SIZE = (64, 80)
    UPSCALE_FACTOR = 8
    NUM_EPOCHS = opt_parser.num_epochs
    turn = 0
    # out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    out_sample_path = 'training_results_sample/SRF_' + \
        str(UPSCALE_FACTOR) + '/'
    epochs_path = 'epochs_%d' % UPSCALE_FACTOR
    csv_path = 'csv_%d' % UPSCALE_FACTOR

    # data_path = './../../../kaist_dataset/kaist-cvpr15/images'
    data_path = opt_parser.data_path

    print("Start getting data")
    train_set = TrainDataset(data_path=data_path,
                             mode='train',
                             lr_size=LR_SIZE)
    val_set = TrainDataset(data_path=data_path,
                           mode='test',
                           lr_size=LR_SIZE)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4,
                            batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel()
          for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel()
          for param in netD.parameters()))

    generator_criterion = GeneratorWLoss()

    if os.path.exists(epochs_path):
        pre_weight = len(os.listdir(epochs_path)) / 2
        netG.load_state_dict(torch.load(epochs_path + '/' + 'netG_epoch_%d_%d.pth' %
                                        (UPSCALE_FACTOR, pre_weight)))
        netD.load_state_dict(torch.load(epochs_path + '/' + 'netD_epoch_%d_%d.pth' %
                                        (UPSCALE_FACTOR, pre_weight)))


    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print("Cuda available")
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    # optimizerG = optim.Adam(netG.parameters())
    # optimizerD = optim.Adam(netD.parameters())
    optimizerG = optim.RMSprop(
        netG.parameters(), lr=opt.lr)
    optimizerD = optim.RMSprop(
        netD.parameters(), lr=opt.lr)

    results = {'d_loss': [], 'g_loss': [], 'd_score': [],
               'g_score': [], 'psnr': [], 'ssim': [], 'psnr_edit': [] }

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0,
                           'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, edge, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = Variable(target)
            edge = Variable(edge)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
                edge = edge.cuda()
            # data = torch.cat((data, edge), dim=1)
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            netD.zero_grad()

            # turn = turn + 1
            # if turn % opt.n_critic == 1:
            real_out = netD(real_img).mean()
            fake_img = netG(z, edge)
            fake_out = netD(fake_img).mean()
            d_loss = fake_out - real_out
            d_loss.backward(retain_graph=True)

            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            turn = turn + 1
            if turn % opt.n_critic == 1:
                netG.zero_grad()
                ## The two lines below are added to prevent runetime error in Google Colab ##
                fake_img = netG(z, edge)
                fake_out = netD(fake_img).mean()

                g_loss = generator_criterion(fake_out, fake_img, real_img)

                g_loss.backward()
                optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] /
                running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        # # out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        # out_sample_path = 'training_results_sample/SRF_' + \
        #     str(UPSCALE_FACTOR) + '/'
        # epochs_path = 'epochs'
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        if not os.path.exists(out_sample_path):
            os.makedirs(out_sample_path)
        if not os.path.exists(epochs_path):
            os.makedirs(epochs_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0,
                              'psnr': 0, 'ssim': 0, 'batch_sizes': 0, 'psnr_edits': 0, 'psnr_edit': 0}
            val_images = []
            val_index = 1
            for val_lr, val_edge, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                # lr = torch.cat((val_lr, val_edge), dim=1)
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                    val_edge = val_edge.cuda()
                sr = netG(lr, val_edge)

                # sr = (sr + 1.0) / 2.0
                # hr = (hr + 1.0) / 2.0
                # sua ve mien 0 1

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (
                    valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / \
                    valing_results['batch_sizes']

                ### Edited ###
                batch_mse_edit = ((sr - hr) ** 2).data.mean()
                valing_results['psnr_edits'] += 10 * \
                    log10((hr.max()**2) / (batch_mse_edit)) * batch_size 
                valing_results['psnr_edit'] = valing_results['psnr_edits'] / valing_results['batch_sizes']
                ######
                
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f PSNR_edit: %.4f' % (
                        valing_results['psnr'], valing_results['ssim'], valing_results['psnr_edit'] ))

                if(val_index == 1):
                    val_images.extend([display_transform()(
                        sr[0]), display_transform()(hr[0])])
                val_index += 1

            val_images = torch.stack(val_images)   # @Thuan: Nimage(40)*c*h*w
            # @Thuan: split to 4 tensors (N/4)*c*h*w
            val_images = torch.chunk(val_images, val_images.size(0) // 2)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=2, padding=5)
                # utils.save_image(
                #     image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                epoch_index = len(os.listdir(out_sample_path)) + 1
                utils.save_image(
                    image, out_sample_path + 'epoch_%d_index_%d.png' % (epoch_index, index), padding=5)
                index += 1

        # save model parameters
        weight_index = len(os.listdir(epochs_path)) / 2 + 1
        torch.save(netG.state_dict(), 'epochs_%d/netG_epoch_%d_%d.pth' %
                   (UPSCALE_FACTOR, UPSCALE_FACTOR, weight_index))
        torch.save(netD.state_dict(), 'epochs_%d/netD_epoch_%d_%d.pth' %
                   (UPSCALE_FACTOR, UPSCALE_FACTOR, weight_index))

        # save loss\scores\psnr\ssim

        results['d_loss'].append(
            running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(
            running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(
            running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(
            running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        results['psnr_edit'].append(valing_results['psnr_edit'])
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim'], 'PSNR_edit': results['psnr_edit']},
            index=range(1, epoch + 1))
        csv_index = len(os.listdir(csv_path)) + 1
        data_frame.to_csv(csv_path + '/'
                          'training_results_epoch_%d.csv' % (csv_index), index_label='Epoch')
