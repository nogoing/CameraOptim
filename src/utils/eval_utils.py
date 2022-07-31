from math import log10, sqrt
from PIL import Image
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage.metrics import structural_similarity



def psnr(reference, compressed):
    reference_np = np.array(reference)
    compressed_np = np.array(compressed)
    mse = np.mean((reference_np - compressed_np) ** 2)
    
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    
    return psnr


def lpips(reference, compressed):
    reference_tensor = torch.from_numpy(np.array(reference)).permute(2, 0, 1) /255.        # 0. ~ 1.
    compressed_tensor = torch.from_numpy(np.array(compressed)).permute(2, 0, 1) /255.    # 0. ~ 1.
    reference_tensor = 2*reference_tensor - 1.      # -1. ~ 1.
    compressed_tensor = 2*compressed_tensor - 1.  # -1. ~ 1. 
    
    if reference_tensor.dim() == 3:
        reference_tensor = reference_tensor.unsqueeze(0)
        compressed_tensor = compressed_tensor.unsqueeze(0)
      
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    
    lpips = lpips_fn(reference_tensor, compressed_tensor)
    
    return lpips.item()


def ssim(reference, compressed):
    reference_np = np.array(reference.convert('L'))
    compressed_np = np.array(compressed.convert('L'))
    
    ssim, _ = structural_similarity(reference_np, compressed_np, full=True)
    
    return ssim