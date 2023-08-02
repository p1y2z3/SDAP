import os
import argparse
import torch
from utils.utils import batch_SSIM, batch_PSNR
from utils.loader import DatasetFromFolder_test
from model.DBSNl import DBSNl
from torchvision import utils as vutils

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--val_path', type=str, default="./dataset/sidd_val_img/SIDD_noisy", 
                    help='validating dataset path')
parser.add_argument('--save_img', type=bool, default=True, help='if save the denoised image')
parser.add_argument('--eval', type=bool, default=True, help='if calculate psnr/ssim')
parser.add_argument('--result_path', type=str, default='./results', 
                    help='path for saving denoised images')
parser.add_argument('--model_path', type=str, default='./ckpt/SDAP_S_for_SIDD.pth')
parser.add_argument('--Enhancement', type=bool, default=True, help='if use performance enhancement')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        input = input.to(device)
        
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
            if args.eval:
                name = img_name[0] + '_{:.2f}_{:.4f}_'.format(psnr_iter,ssim_iter) + '.png'
            else:
                name = img_name[0] + '.png'
            vutils.save_image(im_denoise, os.path.join(args.result_path, name))

    psnr = psnr / img_nums
    ssim = ssim / img_nums
    print('PSNR: {:4.4f}, SSIM: {:5.4f}, Total images: {}'.format(psnr, ssim, img_nums))
    return psnr,ssim



if __name__ == '__main__':

    model = DBSNl().to(device)
    dataset_val =  DatasetFromFolder_test(args.val_path, eval=args.eval, save_img=args.save_img)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=0,
                                                 drop_last=True)    

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    if args.save_img:
        if not os.path.isdir(args.result_path):
            os.makedirs(args.result_path)
    
    print('=> model_path: {:s}'.format(args.model_path))
    print('=> start testing')
    test(dataloader_val, model)




