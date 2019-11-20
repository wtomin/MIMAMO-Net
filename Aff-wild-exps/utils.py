# -*- coding: utf-8 -*-
import os
import sys
import six
import torch
from os.path import join as pjoin
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import numpy as np
import math
from PIL import Image
from torch.nn import functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
def load_module_2or3(model_name, model_def_path):
    """Load model definition module in a manner that is compatible with
    both Python2 and Python3

    Args:
        model_name: The name of the model to be loaded
        model_def_path: The filepath of the module containing the definition

    Return:
        The loaded python module."""
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod
def load_model(model_name, MODEL_DIR):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    model_def_path = pjoin(MODEL_DIR, model_name + '.py')
    weights_path = pjoin(MODEL_DIR, model_name + '.pth')
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net
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
def get_device(device='cuda:0'):
    assert isinstance(device, str)
    num_cuda = torch.cuda.device_count()

    if 'cuda' in device:
        if num_cuda > 0:
            # Found CUDA device, use the GPU
            return torch.device(device)
        # Fallback to CPU
        print('No CUDA devices found, falling back to CPU')
        device = 'cpu'

    if not torch.backends.mkl.is_available():
        raise NotImplementedError(
            'torch.fft on the CPU requires MKL back-end. ' +
            'Please recompile your PyTorch distribution.')
    return torch.device('cpu')
def make_grid_coeff(coeff, normalize=True):
    '''
    Visualization function for building a large image that contains the
    low-pass, high-pass and all intermediate levels in the steerable pyramid. 
    For the complex intermediate bands, the real part is visualized.
    
    Args:
        coeff (list): complex pyramid stored as list containing all levels
        normalize (bool, optional): Defaults to True. Whether to normalize each band
    
    Returns:
        np.ndarray: large image that contains grid of all bands and orientations
    '''
    M, N = coeff[1][0].shape
    Norients = len(coeff[1])
    out = np.zeros((M * 3 - coeff[-1].shape[0], Norients * N *2))
    currentx, currenty = 0, 0
    m, n = coeff[0].shape
    out[currentx: currentx+m, currenty: currenty+n] = 255 * coeff[0]/coeff[0].max()
    currentx, currenty = m, 0
    for i in range(1, len(coeff[:-1])):
        for j in range(len(coeff[1])):
            tmp_real = coeff[i][j].real
            tmp_imag = coeff[i][j].imag
            phase = np.arctan2(tmp_imag,tmp_real)
            amp = np.sqrt(np.power(tmp_imag,2) + np.power(tmp_real, 2))
            m, n = tmp_real.shape
            if normalize:
                amp = 255*(amp-amp.min())/(amp.max()-amp.min())
                phase = 255*(phase - phase.min())/(phase.max()-phase.min())
            amp[m-1,:] = 255
            amp[:,n-1]=255
            phase[m-1,:]=255
            phase[:,n-1] = 255
            out[currentx:currentx+m, currenty:currenty+n] = amp
            out[currentx:currentx+m, currenty+n:currenty+2*n] = phase
            currenty += 2*n
        currentx += m
        currenty = 0

    m, n = coeff[-1].shape
    out[currentx: currentx+m, currenty: currenty+n] = 255 * coeff[-1]/coeff[-1].max()
    out[0,:] = 255
    out[:,0] = 255
    return out.astype(np.uint8)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
#def show_image_3D(image_2d, save_PATH=None):
#    if save_PATH is not None:
#        if not os.path.isdir(os.path.dirname(save_PATH)):
#            os.makedirs(os.path.dirname(save_PATH))
#    M,N = image_2d.shape[:2]
#    X, Y = range(1, M+1), range(1, N+1)
#    Xm, Ym = np.meshgrid(X, Y)
#    fig = plt.figure()
#    ax = Axes3D(plt.gcf())
#    surf = ax.plot_surface(Xm, Ym, image_2d, cmap=cm.coolwarm,)
#    fig.colorbar(surf, shrink=0.5, aspect=10)
#    if save_PATH is not None:
#        plt.savefig(save_PATH)
#    plt.show()
#def windowing_batch(img_batch, device=None): #reference: https://blogs.mathworks.com/steve/2009/12/04/fourier-transform-visualization-using-windowing/
#    M,N  =img_batch.size()[-2:]
#    w1 = np.expand_dims(np.cos(np.linspace(-np.pi/2, np.pi/2, M)), axis=0)
#    w2 = np.expand_dims(np.cos(np.linspace(-np.pi/2, np.pi/2, N)), axis=0)
##    np.copyto(w1, 0.5, where=w1>1/2.)
##    w1 = w1*2
##    np.copyto(w2, 0.5, where=w2>1/2.)
##    w2  = w2*2
#    window = np.dot(w1.T, w2)
#    window = torch.from_numpy(window).type('torch.FloatTensor').to(device)
#    img_batch= torch.mul(img_batch, window)
#    return img_batch
#def extract_phase_mag_from_coeff(coeff):
#    '''
#    Extracting the phase and magnitude for all intermediate levels in the steerable pyramid. 
#    
#    Args:
#        coeff (list): complex pyramid stored as list containing all levels
#    
#    Returns:
#        torch.Tensor (list): magnitude and phase are stacked in the same dimension
#    '''
#    bs, M, N, _ = coeff[1][0].shape
#    inter_coeff = coeff[1:-1]
#    returns = []
#    for coeff_level in inter_coeff:
#        new_coeff_level = []
#        for subband in coeff_level:
#            subband_ = torch.unbind(subband, -1)
#            subband_real, subband_imag = subband_
#            # computing local phase at each scale and orientation
#            subband_phase = torch.atan2(subband_imag, subband_real) # -pi to pi
#            subband_mag = torch.sqrt(torch.pow(subband_real,2)+torch.pow(subband_imag,2))
#            subband_polar = torch.stack((subband_mag, subband_phase),-1)
#            new_coeff_level.append(subband_polar)
#        returns.append(new_coeff_level)
#    returns.insert(0, coeff[0]) # high pass response real
#    returns.append(coeff[-1]) # low pass response
#    return returns
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
    
def amplitude_based_gaussian_blur(mag, phase, g_kernel):
    bs, n_frames, W, H = phase.size()
    mag_phase = torch.mul(mag, phase)
    in_channel = n_frames
    out_channel = n_frames
    m, n = g_kernel.size()
    filters = torch.stack([g_kernel]*out_channel, dim=0)
    filters = torch.unsqueeze(filters, 1) # (output_channel, input_channel/groups, W, H)
    filters =filters.type('torch.FloatTensor').cuda(async=True) if phase.is_cuda else filters
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
class Steerable_Pyramid_Phase(object):
    def __init__(self, height=5, nbands=4, scale_factor=2, device=None, extract_level=1, visualize=False):
        self.pyramid = SCFpyr_PyTorch(
            height=height, 
            nbands=nbands,
            scale_factor=scale_factor, 
            device=device
        )
        self.height = height
        self.nbands = nbands
        self.scale_factor = scale_factor
        self.device = device
        self.extract_level = extract_level
        self.visualize = visualize
    def build_pyramid(self, im_batch, symmetry = True):
        """
        input image batch has 4 dimensions: batch size,  number of phase images, W, H
        """
        bs, num_phase_frames, W, H =im_batch.size()
        trans_im_batch = im_batch.view(bs*num_phase_frames, 1, W, H) # the second dim is 1, indicating it's grayscale image
        if symmetry:
            trans_im_batch = symmetric_extension_batch(trans_im_batch)
        #tic= time()
        coeff_batch = self.pyramid.build(trans_im_batch)
        #print("process {} images for {}".format(bs*num_phase_frames, time()-tic))
        if not isinstance(coeff_batch, list):
            raise ValueError('Batch of coefficients must be a list')

        if self.visualize :
            example_id = 10 # the 10th image from number of phase images
            example_coeff = extract_from_batch(coeff_batch , example_id, symmetry)
            example_coeff = make_grid_coeff(example_coeff)
            example_coeff = Image.fromarray(example_coeff)
            example_img = trans_im_batch[example_id,0,...].cpu().numpy()
            example_img = Image.fromarray(255*example_img/example_img.max())
            example_img.show()
            example_img_remove_symm = trans_im_batch[example_id,0,...].cpu().numpy()
            example_img_remove_symm = 255*example_img_remove_symm/example_img_remove_symm.max()
            if symmetry:
                W, H = example_img_remove_symm.shape
                example_img_remove_symm = example_img_remove_symm[:W//2, :H//2]
                example_img_remove_symm = Image.fromarray(example_img_remove_symm)
                example_img_remove_symm.show()
            example_coeff.show()
        if isinstance(self.extract_level, int):
            extr_level_coeff_batch = self.extract_coeff_level(self.extract_level, coeff_batch)
            W, H, _  = extr_level_coeff_batch.size()[-3:]
            nbands = extr_level_coeff_batch.size()[0]
            extr_level_coeff_batch = extr_level_coeff_batch.view(nbands, bs, num_phase_frames,  W, H, 2)
            extr_level_coeff_batch = extr_level_coeff_batch.permute(1, 0, 2, 3, 4, 5 ).contiguous()
            if symmetry:
                extr_level_coeff_batch = extr_level_coeff_batch[..., :W//2, :H//2, :]
        elif isinstance(self.extract_level, list):
            extr_level_coeff_batch = []
            for level in self.extract_level:
                level_coeff_batch = self.extract_coeff_level(level, coeff_batch)
                W, H, _  = level_coeff_batch.size()[-3:]
                nbands = level_coeff_batch.size()[0]
                level_coeff_batch = level_coeff_batch.view(nbands, bs, num_phase_frames,  W, H, 2)
                level_coeff_batch = level_coeff_batch.permute(1, 0, 2, 3, 4, 5 ).contiguous()
                if symmetry:
                    level_coeff_batch = level_coeff_batch[..., :W//2, :H//2, :]
                extr_level_coeff_batch.append(level_coeff_batch)
        return extr_level_coeff_batch
    def extract_coeff_level(self, level, coeff_batch):
        extr_level_coeff_batch = coeff_batch[level]
        assert isinstance(extr_level_coeff_batch, list)
        extr_level_coeff_batch = torch.stack(extr_level_coeff_batch, 0)
        return extr_level_coeff_batch
    def extract_phase(self, coeff_batch, return_phase=False, return_both = False):
        """
        coeff batch has dimension: batch size, nbands, number phase frames (17), W, H, 2   (2 is for real part and imaginary part) 
        """
        bs, n_bands, n_phase_frames, W,H,_ = coeff_batch.size()
        trans_coeff_batch = coeff_batch.view(bs*  n_bands* n_phase_frames, W, H, -1)
        real_coeff_batch, imag_coeff_batch = torch.unbind(trans_coeff_batch, -1)
        phase_batch = torch.atan2(imag_coeff_batch, real_coeff_batch)
        mag_batch = torch.sqrt(torch.pow(imag_coeff_batch, 2)+torch.pow(real_coeff_batch, 2))
        phase_batch = phase_batch.view(bs*n_bands, n_phase_frames, W, H)
        EPS = 1e-10
        mag_batch = mag_batch.view(bs*n_bands, n_phase_frames, W, H) +EPS # TO avoid mag==0
        assert (mag_batch<=0.0).nonzero().size(0)==0
        
        # phase unwrap over time
        phase_batch = torch_unwrap(phase_batch, discont = math.pi, dim=-3)
        # phase denoising (amplitude-based gaussian blur)
        g_kernel = torch.from_numpy(gaussian_kernel(std=2, tap=11))
        #denoised_phase_batch = amplitude_based_gaussian_blur_numpy(mag_batch, phase_batch, g_kernel)
        denoised_phase_batch = amplitude_based_gaussian_blur(mag_batch, phase_batch, g_kernel)
        denoised_phase_batch = denoised_phase_batch.view(bs, n_bands, n_phase_frames, W, H)
        # phase difference 
        phase_difference_batch = torch_diff(denoised_phase_batch, dim=2)
        phase_difference_batch =  phase_difference_batch.view(bs, n_bands, n_phase_frames-1, W, H)
        if self.visualize:
            phase_example = phase_batch.view(bs, n_bands, n_phase_frames, W, H)[0,...]
            mag_example = mag_batch.view(bs, n_bands, n_phase_frames, W, H)[0,...]
            denoised_phase_example = denoised_phase_batch.view(bs, n_bands, n_phase_frames, W, H)[0,...]
            phase_diff_example = phase_difference_batch.view(bs, n_bands, n_phase_frames-1, W, H)[0,...]
            self.show_3D_subplots(phase_example, title="phase example", first_k_frames=2)
            self.show_3D_subplots(mag_example, title="magnitude example", first_k_frames=2)
            self.show_3D_subplots(denoised_phase_example, title="denoised phase example", first_k_frames=2)
            self.show_3D_subplots(phase_diff_example, title="phase difference example", first_k_frames=2)
        # denoised phase centered
        mean = denoised_phase_batch.mean(-1).mean(-1)
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        denoised_phase_batch = denoised_phase_batch - mean 
        mean = phase_difference_batch.mean(-1).mean(-1)
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        phase_difference_batch = phase_difference_batch - mean
        phase_difference_batch = torch.clamp(phase_difference_batch, -5*math.pi, 5*math.pi)
        if return_both:
            # remove one phase image
            denoised_phase_batch = denoised_phase_batch[:,:,1:, :]
            assert phase_difference_batch.size() == denoised_phase_batch.size()
            result = self.insert_tensors(phase_difference_batch, denoised_phase_batch, dim=2) 
            result = result.cuda()
            return result
        if return_phase:
            return denoised_phase_batch 
        else:
            return phase_difference_batch
    def insert_tensors(self,t_a, t_b, dim):
        size = list(t_a.size())
        size[dim] = 2*size[dim]
        result = torch.zeros(size)
        length = t_a.size(dim)
        for i in range(length):
            slice0 = [slice(None,None)]*len(size)
            slice0[dim]= slice(i, i+1)
            slice1 = [slice(None,None)]*len(size)
            slice1[dim]= slice(i//2, i//2+1)
            if i%2==0:
                result[slice0] = t_a[slice1]
            else:
                result[slice0] = t_b[slice1]
        return result
    def show_3D_subplots(self, data, title, first_k_frames = None):
        """
        data has dimensions: nbands, n_phase_frames, W, H
        """
        nbands, n_phase_frames, W, H = data.size()
        m = nbands
        n = first_k_frames if first_k_frames is not None else n_phase_frames
        
        X, Y = range(1, W+1), range(1, H+1)
        Xm, Ym = np.meshgrid(X, Y)
        for i in range(m):
            fig, ax = plt.subplots(nrows=1, ncols=n, subplot_kw={'projection':"3d"})
            for j in range(n):
                img = data[i, j , ...].cpu().numpy()
                surf = ax[j].plot_surface(Xm, Ym,img , rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=10)
            fig.suptitle(title+": orientation {}".format(i))
        plt.show()
            
# def PadSequence(batch):
#     # Let's assume that "batch" is a list of tuples: (data0, data1, data2, label, flag, names).
#     # data has dimensionality: bs, num_frames, channels, W, H
#     # Sort the batch in the descending order
#     sorted_batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True)
# 	# Get each sequence and pad it
#     num_sequences = len(sorted_batch[0])-1
#     list_seqs = []
#     list_of_padded_seqs = []
#     for i in range(num_sequences):
#         if len(sorted_batch[0][i].size())!=1:
#             sequences = [x[i] for x in sorted_batch ]
#         else:
#             # for label or flag
#             sequences = [x[i].unsqueeze(-1) for x in sorted_batch ]
#         sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).contiguous()
#         list_seqs.append(sequences)
#         list_of_padded_seqs.append(sequences_padded)
#     lengths = torch.LongTensor([len(x) for x in list_seqs[0]])
#     names = [x[-1] for x in sorted_batch] # names, do not need to be padded
#     list_of_padded_seqs.append(names)
#     return list_of_padded_seqs, lengths
            
