from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from steerable.utils import get_device, make_grid_coeff
from utils.phase_utils import *
from PIL import Image
import numpy as np
class Phase_Difference_Extractor(object):
    def __init__(self, height=5, nbands=4, scale_factor=2, 
        extract_level=1, visualize=False):
        '''Phase_Difference_Extractor: A class to do steerable pyramid computation, extract the phase and phase difference
        Usage: 
              build_pyramid(): build complex steerable pyramid coefficients
              extract(): extract phase differences
        Parameters:
            height: int, default 5
                The coefficients levels including low-pass and high-pass
            nbands: int, default 4
                The number of orientations of the bandpass filters
            scale_factor: int, default 2
                Spatial resolution reduction scale scale_factor
            extract_level: int, or list of int numbers, default 1
                If extract_level is an int number, build_pyramid() will only return the coefficients in one level;
                If extract_level is a list, build_pyramid() will only return the coefficients of multiple levels.
            visualize: bool, default False
               If true, the build_pyramid() and extract() will show the processed results.
        '''
    
        self.pyramid = SCFpyr_PyTorch(
            height=height, 
            nbands=nbands,
            scale_factor=scale_factor, 
            device=get_device()
        )
        self.height = height
        self.nbands = nbands
        self.scale_factor = scale_factor
        self.extract_level = extract_level
        self.visualize = visualize
    def build_pyramid(self, im_batch, symmetry = True):
        """
        input image batch has 4 dimensions: batch size, number of phase images, W, H
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
    def extract(self, coeff_batch):
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
        return phase_difference_batch

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