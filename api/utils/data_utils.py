import torchvision.transforms as transforms
import numbers
import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch

class GroupRandomCrop(object):
    def __init__(self, size, seed=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.seed = seed
    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()
        if self.seed is not None:
            random.seed(self.seed)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images

class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, img_group, is_flow=False):
        if self.seed is not None:
            random.seed(self.seed)
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.LANCZOS):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]



class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.stack(img_group, axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=3)
            else:
                return np.stack(img_group, axis=3)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if len(pic.shape)==3:
                W, H, n_frames =pic.shape
                img = torch.from_numpy(pic).permute(2, 0, 1).contiguous() #frames, W, H
            elif len(pic.shape)==4:
                W, H, channel, n_frames = pic.shape
                img = torch.from_numpy(pic).permute(3, 2, 0, 1).contiguous() #frames, channels, W, H
        else:
            raise ValueError("input image has to be ndarray!")
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data

class RandomCrop(object):
    def __init__(self, size, v):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.v = v
    def __call__(self, img):

        w, h = img.size
        th, tw = self.size
        x1 = int(( w - tw)*self.v)
        y1 = int(( h - th)*self.v)
        #print("print x, y:", x1, y1)
        assert(img.size[0] == w and img.size[1] == h)
        if w == tw and h == th:
            out_image = img
        else:
            out_image = img.crop((x1, y1, x1 + tw, y1 + th)) #same cropping method for all images in the same group
        return out_image

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, v):
        self.v = v
        return
    def __call__(self, img):
        if self.v < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT) 
        #print ("horiontal flip: ",self.v)
        return img

def augment_transforms(meta, resize=256, random_crop=True, 
                      override_meta_imsize=False):
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    if random_crop:
        v = random.random()
        transform_list = [transforms.Resize(resize),
                          RandomCrop(im_size[0], v),
                          RandomHorizontalFlip(v)]
    else:
        if override_meta_imsize:
            im_size = (resize, resize)
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1,1,1]: # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)
