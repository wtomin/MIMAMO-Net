import torch
import math
from torch.nn import functional as F
import numpy as np
def torch_unwrap(tensor, discont=math.pi, dim=-1):
    nd = len(tensor.size())
    dd = torch_diff(tensor, dim=dim)
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[dim] = slice(1, None)
    slice1 = tuple(slice1)
    PI = math.pi
    ddmod = torch.fmod(dd + PI, 2*PI) - PI
    id1 = (ddmod == -PI) & (dd > 0)
    ddmod[id1] = PI
    ph_correct = ddmod - dd
    id2 = torch.abs(dd) < discont
    ph_correct[id2] = 0
    up = tensor.clone().detach()
    up[slice1] = tensor[slice1] + ph_correct.cumsum(dim=dim)
    return up
def torch_diff(tensor,n=1, dim=-1):
    """
    tensor : Input Tensor
    n : int, optional
        The number of times values are differenced. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    """
    nd = len(tensor.size())
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[dim] = slice(1, None)
    slice2[dim] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    for _ in range(n):
        tensor = tensor[slice1] - tensor[slice2]
    return tensor
def extract_from_batch(coeff_batch, example_idx=0, symmetry = True):
    '''
    Given the batched Complex Steerable Pyramid, extract the coefficients
    for a single example from the batch. Additionally, it converts all
    torch.Tensor's to np.ndarrays' and changes creates proper np.complex
    objects for all the orientation bands. 
    Args:
        coeff_batch (list): list containing low-pass, high-pass and pyr levels
        example_idx (int, optional): Defaults to 0. index in batch to extract
    
    Returns:
        list: list containing low-pass, high-pass and pyr levels as np.ndarray
    '''
    if not isinstance(coeff_batch, list):
        raise ValueError('Batch of coefficients must be a list')
    coeff = []  # coefficient for single example
    for coeff_level in coeff_batch:
        if isinstance(coeff_level, torch.Tensor):
            # Low- or High-Pass
            coeff_level_numpy = coeff_level[example_idx].cpu().numpy()
            if symmetry:
                W, H = coeff_level_numpy.shape
                coeff_level_numpy = coeff_level_numpy[:W//2, :H//2]
            coeff.append(coeff_level_numpy)
        elif isinstance(coeff_level, list):
            coeff_orientations_numpy = []
            for coeff_orientation in coeff_level:
                coeff_orientation_numpy = coeff_orientation[example_idx].cpu().numpy()
                coeff_orientation_numpy = coeff_orientation_numpy[:,:,0] + 1j*coeff_orientation_numpy[:,:,1]
                if symmetry:
                    W, H = coeff_orientation_numpy.shape
                    coeff_orientation_numpy= coeff_orientation_numpy[:W//2, :H//2]
                coeff_orientations_numpy.append(coeff_orientation_numpy)
            coeff.append(coeff_orientations_numpy)
        else:
            raise ValueError('coeff leve must be of type (list, torch.Tensor)')
    return coeff
def amplitude_based_gaussian_blur(mag, phase, g_kernel):
    bs, n_frames, W, H = phase.size()
    mag_phase = torch.mul(mag, phase)
    in_channel = n_frames
    out_channel = n_frames
    m, n = g_kernel.size()
    filters = torch.stack([g_kernel]*out_channel, dim=0)
    filters = torch.unsqueeze(filters, 1) # (output_channel, input_channel/groups, W, H)
    filters = filters.type(phase.type())
    mag_phase_blurred = F.conv2d(mag_phase, filters, groups = in_channel, padding=m//2)
    mag_blurred = F.conv2d(mag, filters, groups = in_channel, padding=m//2)
    result = torch.div(mag_phase_blurred, mag_blurred)
    return result
def amplitude_based_gaussian_blurcoeff_batch_numpy(mag, phase, g_kernel):
    bs, n_frames, W, H = phase.size()
    phase = phase.cpu().numpy()
    mag = mag.cpu().numpy()
    g_kernel = g_kernel.cpu().numpy()
    from scipy.signal import convolve2d
    new_phase = []
    for b in range(bs):
        new_phase_b = []
        for f in range(n_frames):
            p = phase[b,f,...]
            m = mag[b, f,...]
            denoised_phase =  convolve2d(np.multiply(p, m), g_kernel, mode='same')/convolve2d(m, g_kernel, mode='same')
            new_phase_b.append(denoised_phase)
        new_phase.append(new_phase_b)
    new_phase = np.asarray(new_phase)
    return torch.Tensor(new_phase).type('torch.FloatTensor').cuda(async=True)
def gaussian_kernel(std, tap = 11):
    kernel = np.zeros((tap, tap))
    for x in range(tap):
        for y in range(tap):
            x0 = x - tap//2
            y0 = y - tap//2
            kernel[x, y] = np.exp(- (x0**2+y0**2)/(2*std**2))
    return kernel
def symmetric_extension_batch(img_batch):
    #  img_batch, None, W, H (the last two aixes are two dimensional image)
    img_batch_inverse_col = img_batch.clone().detach()
    inv_idx_col = torch.arange(img_batch.size(-1)-1, -1, -1).long()
    img_batch_inverse_col = img_batch_inverse_col[..., :, inv_idx_col]
    img_batch_inverse_row = img_batch.clone().detach()
    inv_idx_row = torch.arange(img_batch.size(-2)-1, -1, -1).long()
    img_batch_inverse_row = img_batch_inverse_row[..., inv_idx_row, :]
    img_batch_inverse_row_col = img_batch_inverse_col.clone().detach()
    img_batch_inverse_row_col = img_batch_inverse_row_col[:,:, inv_idx_row, :]
    img_batch_0 = torch.cat([img_batch, img_batch_inverse_col], dim=-1)
    img_batch_1 = torch.cat([img_batch_inverse_row, img_batch_inverse_row_col], dim=-1)
    new_img_batch = torch.cat([img_batch_0, img_batch_1], dim=-2)
    return new_img_batch
