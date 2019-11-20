
from __future__ import division

import os
import time
import six
import sys
from tqdm import tqdm
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data
from os.path import join as pjoin
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import glob
import numbers
from PIL import Image, ImageOps
import random
import copy
# for torch lower version
import torch._utils
from torch.nn import functional as F
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
global parsed
import torch.utils.data as data
class VideoRecord(object):
    def __init__(self, video, face_dir, annot_dir, label_name, test_mode = False):
        self.video = video
        self.face_dir = face_dir
        self.annot_dir = annot_dir
        self.label_name = label_name
        self.test_mode = test_mode
        self.path_label = self.get_path_label()
     
    def get_path_label(self):
        frames = glob.glob(os.path.join(self.face_dir, self.video, self.video+"_aligned", '*.bmp'))
        frames = sorted(frames, key  = lambda x: os.path.basename(x).split(".")[0].split("_")[-1])
        if len(frames)==0:
            raise ValueError("number of frames of video {} should not be zero.".format(self.video))
        annot_file = os.path.join(self.annot_dir, 'train',self.label_name, self.video+".txt")
        if (not self.test_mode) and (not os.path.exists(annot_file)):
            raise ValueError("Annotation file not found: the training mode should always has annotation file!")
        if self.test_mode:
            return [frames, [-100]*len(frames)]
        else:
            f = open(annot_file, "r")
            corr_frames, labels = [], []
            for i, x in enumerate(f):
                label = float(x)
                corr_frame = os.path.join(self.face_dir, self.video, self.video+"_aligned", 'frame_det_00_{0:06d}.bmp'.format(i+1))
                if os.path.exists(corr_frame):
                    corr_frames.append(corr_frame)
                    labels.append(label)
                else:
                    # skip those frames and labels
                    continue
            f.close()
            assert len(corr_frames) == len(labels)
            return [corr_frames, labels]
    def __str__(self):
        string = ''
        for key, record in self.utterance_dict.items():
            string += str(record)+'\n'
        return string
 
class Face_Dataset(data.Dataset):
    def __init__(self, root_path, annot_dir, video_name_list,  label_name, test_mode =False, transform = None):
        self.root_path = root_path
        self.annot_dir = annot_dir
        self.video_name_list = video_name_list
        self.label_name = label_name
        self.test_mode = test_mode
        self.transform = transform
        self.parse_videos()

    def parse_videos(self):
        videos = self.video_name_list
        self.video_list = list()
        self.frame_ids = []
        self.video_ids = []
        i = 0
        for vid in tqdm(videos):
            v_record = VideoRecord(vid, self.root_path, self.annot_dir, self.label_name, self.test_mode)
            frames, labels = v_record.path_label
            if len(frames) !=0 and (len(frames)==len(labels)):
                self.video_list.append(v_record)
            self.video_ids.extend([i]*len(frames))
            self.frame_ids.extend(list(np.arange(len(frames))))
            i +=1
        print("number of videos:{}, number of frames:{}".format(len(self.video_list), len(self.frame_ids)))

    def __len__(self):
        return len(self.frame_ids)
    def __getitem__(self, index):
        f_id = self.frame_ids[index]
        video_record = self.video_list[self.video_ids[index]]
        frames, labels = video_record.path_label
        frame, label = frames[f_id], labels[f_id]
        img = Image.open(frame)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, frame, video_record.video
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

def compose_transforms(meta, resize=256, center_crop=True, 
                      override_meta_imsize=False):
    """Compose preprocessing transforms for model

    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.

    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
        override_meta_imsize (bool) [False]: if true, use the value of `resize`
           to select the image input size, rather than the properties contained
           in meta (this option only applies when center cropping is not used.

    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    if center_crop:
        transform_list = [transforms.Resize(resize),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    else:
        if override_meta_imsize:
            im_size = (resize, resize)
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
            
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1,1,1]: # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)

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
def get_vec( model, layer_name, image):
    bs = image.size(0)
    if parsed.layer_name == 'pool5_full':
        layer_name = 'pool5'
    layer = model._modules.get(layer_name)
    if parsed.layer_name == 'fc7':
        layer_output_size = 4096
        my_embedding = torch.zeros(bs, layer_output_size)
    elif parsed.layer_name == 'fc8':
        my_embedding = torch.zeros(bs, 7)
    elif parsed.layer_name == 'pool5' or parsed.layer_name == 'pool5_full':
        my_embedding = torch.zeros([bs, 512, 7, 7])
    elif parsed.layer_name == 'pool4':
        my_embedding = torch.zeros([bs, 512, 14, 14])
    elif parsed.layer_name == 'pool3':
        my_embedding = torch.zeros([bs, 256, 28, 28])
    elif parsed.layer_name =='pool5_7x7_s1':
         my_embedding= torch.zeros([bs, 2048, 1, 1])
    elif parsed.layer_name =='conv5_3_3x3_relu':
         my_embedding = torch.zeros([bs, 512, 7, 7])
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    h_x = model(image)
    h.remove()
    if parsed.layer_name == 'pool5' or parsed.layer_name == 'conv5_3_3x3_relu':
        GAP_layer = nn.AvgPool2d(kernel_size = [7,7], stride=(1, 1))
        my_embedding = GAP_layer(my_embedding)
    return F.relu(my_embedding.squeeze())
def get_frame_index(frame_path):
    frame_name = frame_path.split('/')[-1]
    frame_num  = int(frame_name.split('.')[0].split('_')[-1])
    return frame_num
def predict(data_loader,layer_name, model, des_dir):
    with torch.no_grad():
        for ims, target, img_path, video_name in tqdm(data_loader):
            ims = ims.cuda(async=True)
            output = get_vec(model, layer_name, ims)
            for feature, path, video_n in zip(output, img_path, video_name):
                des_path = os.path.join(des_dir, video_n)
                if not os.path.exists(des_path):
                    os.makedirs(des_path)
                des_path = os.path.join(des_path, "%05d.npy"%get_frame_index(path))
                np.save(des_path, feature)


def feature_extraction(model, loader, des_dir):
    model.eval()
    predict(loader, parsed.layer_name, model, des_dir)
    
def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    MODEL_DIR = '/media/newssd/pytorch-benchmarks/models/fer+'
    model_name = 'resnet50_ferplus_dag'
    model = load_model(model_name, MODEL_DIR)
    try:
        device = torch.device("cuda")
        model = model.to(device)
    except:
        torch.cuda.set_device(0)
        model = model.cuda()
    DATA_DIR = parsed.data_root
    annot_dir = '/media/newssd/Aff-Wild_experiments/annotations'
    video_names = os.listdir(DATA_DIR)
    meta = model.meta
    preproc_transforms = compose_transforms(meta, center_crop=False) if not parsed.augment else augment_transforms(meta, random_crop=True)

    dataset = Face_Dataset(DATA_DIR, annot_dir, video_names, label_name='arousal', test_mode=False if 'train' in DATA_DIR else True, transform=preproc_transforms)
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = 4, 
        num_workers=0, pin_memory=False )
    des_dir = pjoin(parsed.save_root, '_'.join(['{}_features'.format(model_name), 'fps='+str(parsed.fps), parsed.layer_name]))
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    feature_extraction(model, data_loader, des_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run.')           
    parser.add_argument('--refresh', dest='refresh', action='store_true',
                        help='refresh feature cache') 
    parser.add_argument('--fps',type=int, default = 30, 
                        help='frames per second to extract') 
    parser.add_argument('--layer_name',type=str, default = 'pool5_7x7_s1', 
                        help='frames per second to extract') 
    parser.add_argument('--augment', action="store_true", 
                        help='whether to extract augmented features for train set only ') 

    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--data_root', type=str, default='')
    parser.set_defaults(refresh=False)
    parsed = parser.parse_args()
    
    main()
