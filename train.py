import os, time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import AverageMeter, batch_SSIM, batch_PSNR
from utils.loader import DatasetFromFolder_train, DatasetFromFolder_test
from model.DBSNl import DBSNl
import numpy as np
import time
from torchvision import utils as vutils

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--ps', default=160, type=int, help='patch size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--init_epoch', type=int, default=1,
                    help='if finetune model, set the initial epoch')
parser.add_argument('--epochs', default=50, type=int, help='sum of epochs')
parser.add_argument('--train_path', type=str, default="./dataset/train_data",
                    help='training dataset path')
parser.add_argument('--val_path', type=str, default="./dataset/sidd_val_img/SIDD_noisy", 
                    help='validating dataset path')
parser.add_argument('--gpu', type=str, default="1", help='GPU id')
parser.add_argument('--save_img', type=bool, default=True, help='if save the last validated image for comparison')
parser.add_argument('--eval', type=bool, default=True, help='if calculate psnr/ssim')
parser.add_argument('--result_path', type=str, default='./results', 
                    help='path for saving denoised images')
parser.add_argument('--pretrained_model', type=str, default='./ckpt/SDAP.pth',
                     help='training loss')
parser.add_argument('--save_model_dir', type=str, default='./ckpt/experiment1', help='path to save models')
parser.add_argument('--Enhancement', type=bool, default=False, help='if use performance enhancement')

args = parser.parse_args()
psnr_max = 0

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask_pair(img, p):
    # prepare masks (N x C x H/p x W/p)
    n, c, h, w = img.shape
    mask = []
    for i in range(p*p):
        mask.append(torch.zeros(size=(n * h // p * w // p * p*p, ),
                        dtype=torch.bool,
                        device=img.device))

    rd_pair_idx = np.zeros([n * h // p * w // p, p*p], dtype=np.int64)
    for i in range(n * h // p * w // p):
        array = np.arange(p*p)
        np.random.shuffle(array)
        rd_pair_idx[i, :] = array
    rd_pair_idx = torch.from_numpy(rd_pair_idx).cuda()

    rd_pair_idx += torch.arange(start=0,
                                end=n * h // p * w // p * p*p,
                                step=p*p,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    for i in range(p*p):
        mask[i][rd_pair_idx[:, i]] = 1
    return mask


def generate_subimages(img, mask, p):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // p,
                           w // p,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=p)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // p, w // p, 1).permute(0, 3, 1, 2)
    return subimage


def train(dataloader_val, train_loader, model, optimizer, epoch, epoch_total, L1):
    loss_sum = 0
    losses = AverageMeter()
    model.train()
    start_time = time.time()
    p = 5   # RSG stride factor
    global psnr_max

    for i, noise_img in enumerate(train_loader):
        input_var = noise_img.cuda()
        optimizer.zero_grad()
        
        ###########################  RSG  ##############################
        mask = generate_mask_pair(input_var, p)
        n, c, h, w = input_var.size()
        noisy_sub = torch.zeros([p*p,n,c,h//p,w//p]).cuda()
        for j in range(p*p):
            noisy_sub[j,...] = generate_subimages(input_var, mask[j], p)
        #################################################################

        sub_denoised = torch.zeros([p*p,n,c,h//p,w//p]).cuda()
        for j in range(p * p):
            sub_denoised[j,...] = model(noisy_sub[j,...])

        loss = 0
        for j in range(p*p):
            loss += L1(sub_denoised[j,...], noisy_sub[(j + 1) % (p*p),...])

        loss_sum+=loss.item()
        losses.update(loss.item())

        loss.backward()
        optimizer.step()
        if (i % 10 == 0) and (i != 0):
            loss_avg = loss_sum / 10
            loss_sum = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f} Time: {:4.4f}s".format(
                epoch, epoch_total, i, len(train_loader), loss_avg, time.time() - start_time))
            start_time = time.time()

        if (i % 300 == 0 or i==len(train_loader)-1) and (i != 0):
            psnr, ssim = test(dataloader_val, model)
            torch.save(model.state_dict(), os.path.join(args.save_model_dir,
                                            'checkpoint_epoch_{:0>4}_{}_{:.2f}_{:.4f}.pth'.format(epoch, i, psnr, ssim)))
            if psnr > psnr_max:
                psnr_max = max(psnr, psnr_max)
                torch.save(model.state_dict(), os.path.join(args.save_model_dir,
                                            'checkpoint_best.pth'.format(epoch, i, psnr, ssim)))
            start_time = time.time()

    return losses.avg


def pixel_unshuffle(input, factor):
    """
    (n, c, h, w) ===> (n*factor^2, c, h/factor, w/factor)
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // factor
    out_width = in_width // factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, factor,
        out_width, factor)

    batch_size *= factor ** 2
    unshuffle_out = input_view.permute(0, 3, 5, 1, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

def pixel_shuffle(input, factor):
    """
    (n*factor^2, c, h/factor, w/factor) ===> (n, c, h, w)
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height * factor
    out_width = in_width * factor

    batch_size /= factor ** 2
    batch_size = int(batch_size)
    input_view = input.contiguous().view(
        batch_size, factor, factor, channels, in_height,
        in_width)

    unshuffle_out = input_view.permute(0, 3, 4, 1, 5, 2).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


def test(dataloader_val, model):
    psnr = 0
    ssim = 0
    model.eval()
    img_nums = len(dataloader_val)
    for i, data in enumerate(dataloader_val):
        if args.eval:
            if args.save_img:
                input, label, img_name = data
            else:
                input, label = data
        else:
            if args.save_img:
                input, img_name = data
            else:
                input = data

        input = input.cuda()
        
        with torch.no_grad():
            input = pixel_unshuffle(input, 2)
            test_out = model(input)
            im_denoise = pixel_shuffle(test_out,2)
            if args.Enhancement:
                im_denoise = model(im_denoise)

        im_denoise.clamp_(0.0, 1.0)

        if args.eval:
            psnr_iter = batch_PSNR(im_denoise, label)
            ssim_iter = batch_SSIM(im_denoise, label)
            psnr+=psnr_iter
            ssim+=ssim_iter

        if args.save_img:
            name = img_name[0] + '.png'
            vutils.save_image(im_denoise, os.path.join(args.result_path, name))

    psnr = psnr / img_nums
    ssim = ssim / img_nums
    print('Validating: {:0>3} , PSNR: {:4.4f}, SSIM: {:5.4f}, PSNR_max: {:4.4f}'.format(img_nums, psnr, ssim, psnr_max))
    return psnr,ssim

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    assert args.ps % 5 == 0, 'patch size must be set to a multiple of 5.'
  
    if not os.path.isdir(args.save_model_dir):
            os.makedirs(args.save_model_dir)

    if args.save_img:
        if not os.path.isdir(args.result_path):
            os.makedirs(args.result_path)

    model = DBSNl().cuda()
    criterionL1 = nn.L1Loss().cuda()

    if os.path.exists(args.pretrained_model):
        # load existing model
        print("=> loading model '{}'".format(args.pretrained_model))
        model.load_state_dict(torch.load(args.pretrained_model))
        cur_epoch = args.init_epoch
    else:
        cur_epoch = 0
        print("=> no model found at '{}'".format(args.pretrained_model))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)        

    print('=> load dataset')
    train_dataset = DatasetFromFolder_train(args.train_path,patch_size = args.ps)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    dataset_val =  DatasetFromFolder_test(args.val_path, eval=args.eval, save_img=args.save_img)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=0,
                                                 drop_last=True)
    
    print('=> start training')
    for epoch in range(cur_epoch, args.epochs + 1):
        loss = train(dataloader_val, train_loader, model, optimizer, epoch, args.epochs + 1,criterionL1)
        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}'
            .format(
            epoch,
            lr=optimizer.param_groups[-1]['lr'],
            loss=loss))
