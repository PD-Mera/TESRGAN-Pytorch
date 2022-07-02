import os
from math import log10

import pandas as pd
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TrainDataset, display_transform
from model import Generator
from ssim import ssim


if __name__ == '__main__':
    UPSCALE_FACTOR = 4
    out_sample_path = 'val_results_sample/SRF_' + \
        str(UPSCALE_FACTOR) + '/'

    epochs_path = 'epochs'
    csv_path = 'csv'
    start_epoch = 38
    stop_epoch = 51

    LR_SIZE = (128, 160)

    data_path = './../../../kaist_dataset/kaist-cvpr15/images'

    val_set = TrainDataset(data_path=data_path,
                           mode='val_all_set',
                           lr_size=LR_SIZE)
    val_loader = DataLoader(dataset=val_set, num_workers=4,
                            batch_size=2, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        netG.cuda()

    results = {'psnr': [], 'ssim': []}

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for epoch in range(start_epoch, stop_epoch):
        if os.path.exists(epochs_path):
            netG.load_state_dict(torch.load(epochs_path + '/' + 'netG_epoch_%d_%d.pth' %
                                            (UPSCALE_FACTOR, epoch)))

        netG.eval()

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0,
                              'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            val_index = 1
            for val_lr, val_edge, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = torch.cat((val_lr, val_edge), dim=1)
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr).item()

                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (
                    valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / \
                    valing_results['batch_sizes']
                val_bar.set_description(
                    desc='Epoch %d: [converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        epoch, valing_results['psnr'], valing_results['ssim']))

                if(val_index == 1):
                    val_images.extend([display_transform()(
                        sr[0]), display_transform()(hr[0])])
                val_index += 1

            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 2)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=2, padding=5)
                epoch_index = len(os.listdir(out_sample_path)) + 1
                utils.save_image(
                    image, out_sample_path + 'epoch_%d_index_%d.png' % (epoch_index, index), padding=5)

        # save model parameters
        weight_index = len(os.listdir(epochs_path)) / 2 + 1
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' %
                   (UPSCALE_FACTOR, weight_index))

        # save loss\scores\psnr\ssim
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        data_frame = pd.DataFrame(
            data={'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(start_epoch, epoch + 1))
        csv_index = len(os.listdir(csv_path)) + 1
        data_frame.to_csv(csv_path + '/'
                          'val_results_epoch_%d.csv' % csv_index, index_label='Epoch')
