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


if __name__ == "__main__":
    trans = torchvision.transforms.Compose([
        GroupScale(256),
        GroupRandomCrop(224),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )]
    )

    im = Image.open('../tensorflow-model-zoo.torch/lena_299.png')

    color_group = [im] * 3
    rst = trans(color_group)

    gray_group = [im.convert('L')] * 9
    gray_rst = trans(gray_group)

    trans2 = torchvision.transforms.Compose([
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225])
    ])
    print(trans2(color_group))
