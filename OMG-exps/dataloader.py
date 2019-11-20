import torch.utils.data as data
from collections import OrderedDict
from itertools import islice
import pdb 
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pickle
import glob
from copy import copy
import json
import torch
from torch import nn as nn
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from transforms import GroupRandomCrop,GroupRandomHorizontalFlip, GroupNormalize, GroupScale, Stack, ToTorchFormatTensor
import torchvision
from utils import Steerable_Pyramid_Phase, get_device
from random import shuffle
class UtteranceRecord(object):
    def __init__(self, video, utter, root_path, data_row, label_name, mini_num_frames = 17):
        self.video = video
        self.utterance = utter if 'mp4' not in utter else utter.split(".")[0]
        self.mini_num_frames = mini_num_frames
        if not "_" in label_name:
            self._label_name = label_name
        else:
            self._label_name = label_name.split("_")
        self.root_path = root_path
        self.data_row = data_row
    @property
    def path(self):
        #paths includes all frames (has face)
        if isinstance(self.root_path, list):
            frames = sorted(glob.glob(os.path.join(self.root_path[0], self.video, self.utterance, self.utterance+"_aligned", '*.bmp')), key = lambda x: int(x.split(".")[0].split("_")[-1]))
            if len(frames)==0:
                frames = sorted(glob.glob(os.path.join(self.root_path[1], self.video, self.utterance, self.utterance+"_aligned", '*.bmp')), key = lambda x: int(x.split(".")[0].split("_")[-1]))
        else:
            frames = sorted(glob.glob(os.path.join(self.root_path, self.video, self.utterance, self.utterance+"_aligned", '*.bmp')), key = lambda x: int(x.split(".")[0].split("_")[-1]))
        if len(frames) >=self.mini_num_frames: # minimum number of frames to extract Phase
            return frames
        else:
            return None
    @property
    def label(self):
        if isinstance(self._label_name, list):
            return np.array([self.data_row[label_name] for label_name in self._label_name])
        elif isinstance(self._label_name, str):
            label = self.data_row[self._label_name]
            return label
    @property
    def name(self):
        return "{} {}".format(self.video, self.utterance)
    def __str__(self):
        return "{} in {}, {}: {}, number of facial frames:{}".format(self.utterance, self.video, self._label_name, self.label, len(self.path))
def phase_2_output( phase_batch, steerable_pyramid,return_phase=False):
    """
    phase_batch dim: bs, num_phase, W, H
    """
    sp = steerable_pyramid
    num_frames,num_phases, W, H = phase_batch.size()
    coeff_batch = sp.build_pyramid(phase_batch)
    assert isinstance(coeff_batch, list)
    phase_batch_0 = sp.extract_phase(coeff_batch[0], return_phase=return_phase)
    num_frames, n_ch, n_ph, W, H= phase_batch_0.size()
    phase_batch_0 = phase_batch_0.view(num_frames, -1, W, H)
    phase_batch_1 = sp.extract_phase(coeff_batch[1], return_phase=return_phase)
    num_frames, n_ch, n_ph, W, H= phase_batch_1.size()
    phase_batch_1 = phase_batch_1.view(num_frames, -1, W, H)
    return phase_batch_0,phase_batch_1
class Face_Dataset(data.Dataset):
    def __init__(self, root_path, pretrained_feature_root, dict_file, label_name, py_level, py_nbands,  
                 sample_rate=4, num_phase = 8, phase_size = 112, test_mode =False, return_phase=False,
                  max_len = 16): # the length distribution: <13s many, 16s: 2, 18s: 1, 19s:1, 23s:1, 24s:1
        """
        Args: 
            root_path: the directory that contains all facial images
            dict_file: data dictory
            label_name: arousal or valence or arousal_valence
            max_len: max sequence length for frames in one video (including many utterances)
            sample_rate: the sample rate for RGB images
            num_phase: number of phase images (to be extracted for phase and phase difference)
            phase_size: phase image size
            test_mode: for training, True, otherwise it's False
        """
        self.root_path = root_path
        self.pretrained_feature_root = pretrained_feature_root
        self.dict_file= dict_file
        self.label_name = label_name # one of 'arousal', 'valence' and 'EmotionMaxVote'
        self.test_mode = test_mode
        self.sample_rate = sample_rate # frames/second
        self.max_len = max_len*self.sample_rate
        self.fps=30
        self.num_phase = num_phase
        self.return_phase = return_phase
        self._parse_dict()
        self.utterance_list = self.sample_images(self.utterance_list)
        print("total number of utterances:{}".format(len(self.utterance_list)))
        self.phase_size = phase_size
        device = get_device('cuda:0')
        self.steerable_pyramid = Steerable_Pyramid_Phase(height=py_level, nbands=py_nbands, scale_factor=2, device=device, extract_level=[1,2], visualize=False)
        if not self.test_mode:
            shuffle(self.utterance_list) 
        seq_lens = [record.length for record in self.utterance_list ]
        unique_seq_lens = np.unique(seq_lens)
        lens_index_list= []
        for length in unique_seq_lens:
            indexes = [id for id, record in enumerate(self.utterance_list) if record.length==length]
            lens_index_list.append(indexes)
        self.indices_list =  lens_index_list
    def _parse_dict(self):
        data_dict = pickle.load(open(self.dict_file,'rb')) if isinstance(self.dict_file, str) else self.dict_file 
        self.utterance_list = list()
        num_videos = 0
        videos = list(data_dict.keys())
        for id, video in tqdm(enumerate(videos)):
            num_videos+=1
            for utter in data_dict[video].keys():
                u_record = UtteranceRecord(video, utter,self.root_path, data_dict[video][utter], self.label_name, mini_num_frames=self.num_phase+1)
                if u_record.path is not None:
                    self.utterance_list.append(u_record)
        print("total number of videos:{}".format(num_videos)) 
    def sample_images(self, utterance_list):
        new_utterance_list= []
        for utter_record in tqdm(utterance_list):
            total_frames = utter_record.path
            n_frames = self.fps//self.sample_rate
            if not self.test_mode: # train set has augmentation on sampling image
                max_aug = min(n_frames, len(total_frames))
                augment_utterance_list = []
                for i in range(max_aug-1):
                    frames  = total_frames[i:]
                    sampled_rgb_frames = []
                    sampled_phase_frames = []
                    sampled_rgb_f_index = np.arange(len(frames))[::n_frames]
                    while len(frames) - sampled_rgb_f_index[-1]<self.num_phase+1:
                        sampled_rgb_f_index = sampled_rgb_f_index[:-1] # remove the last rgb frame
                        if len(sampled_rgb_f_index)==0:
                            break
                    if len(sampled_rgb_f_index)==0:
                        break
                    sampled_phase_f_index = np.array([np.arange(id, id+self.num_phase+1) for id in sampled_rgb_f_index])
                    sampled_rgb_frames.extend([frames[id + self.num_phase//2] for id in sampled_rgb_f_index])
                    sampled_phase_frames.extend([np.array([frames[id] for id in ids]) for ids in sampled_phase_f_index]) 
                    length = min(len(sampled_rgb_frames), self.max_len) 
                    if length >0: # make sure that record in augment_utterance_list has the same length
                        setattr(utter_record, 'rgb_frames', sampled_rgb_frames[:length])
                        setattr(utter_record, 'phase_frames', sampled_phase_frames[:length])
                        setattr(utter_record, 'length', len(sampled_phase_frames[:length]))
                        augment_utterance_list.append(utter_record)
                if len(augment_utterance_list)!=0: 
                    augment_utterance_list = augment_utterance_list[::3] # save training time
                    new_utterance_list.extend(augment_utterance_list)
                else:
                    print("Nothing in augment list")
            else:
                frames = total_frames
                sampled_rgb_frames = []
                sampled_phase_frames = []
                sampled_rgb_f_index = np.arange(len(frames))[::n_frames]
                while len(frames) - sampled_rgb_f_index[-1]<self.num_phase+1:
                    sampled_rgb_f_index = sampled_rgb_f_index[:-1] # remove the last rgb frame
                    if len(sampled_rgb_f_index)==0:
                        break
                if len(sampled_rgb_f_index)==0:
                    raise ValueError("Frame length incorrect")
                sampled_phase_f_index = np.array([np.arange(id, id+self.num_phase+1) for id in sampled_rgb_f_index])
                sampled_rgb_frames.extend([frames[id + self.num_phase//2] for id in sampled_rgb_f_index])
                sampled_phase_frames.extend([np.array([frames[id] for id in ids]) for ids in sampled_phase_f_index]) 
                length = min(len(sampled_rgb_frames),self.max_len) 
                if length>0:
                    setattr(utter_record, 'rgb_frames', sampled_rgb_frames[:length])
                    setattr(utter_record, 'phase_frames', sampled_phase_frames[:length])
                    setattr(utter_record, 'length', len(sampled_phase_frames[:length]))
                    new_utterance_list.append(utter_record)
        return new_utterance_list
                        
    def __getitem__(self, index):
        u_record = self.utterance_list[index]
        rgb_frames = u_record.rgb_frames
        phase_frames = u_record.phase_frames
        label = u_record.label
        while len(rgb_frames)==0:
            id = np.random.randint(len(self.utterance_list))
            u_record = self.utterance_list[id] 
            rgb_frames = u_record.rgb_frames
            phase_frames = u_record.phase_frames

        name = u_record.name
        return_list = self.get(rgb_frames, phase_frames)
        return_list.append(label)
        return_list.append(name)
        return return_list     
    def get(self, rgb_frames, phase_frames):
        assert len(rgb_frames) == len(phase_frames) 
        assert len(rgb_frames)<=self.max_len  
        phase_images = []
        for frames in phase_frames:
            phase_img_list = []
            for frame in frames:
                img = Image.open(frame).convert('L')
                phase_img_list.append(img)
            phase_images.append(phase_img_list)

        if not self.test_mode:
            random_seed = np.random.randint(250)
            W,H = phase_images[0][0].size
            phase_transform = torchvision.transforms.Compose([GroupRandomHorizontalFlip(seed=random_seed),
                                   GroupRandomCrop(size=int(W*0.85), seed=random_seed),
                                   GroupScale(size=self.phase_size),
                                   Stack(),
                                   ToTorchFormatTensor()])
        else:
            phase_transform = torchvision.transforms.Compose([
                                   GroupScale(size=self.phase_size),
                                   Stack(),
                                   ToTorchFormatTensor()])   

        flat_phase_images =[]
        for sublist in phase_images:
            flat_phase_images.extend(sublist)
        flat_phase_images_trans = phase_transform(flat_phase_images)
        phase_images = flat_phase_images_trans.view(len(phase_images), self.num_phase+1, self.phase_size, self.phase_size)
        phase_images = phase_images.type('torch.FloatTensor').cuda()
        phase_batch_0,phase_batch_1 = phase_2_output( phase_images, self.steerable_pyramid, return_phase=self.return_phase)  
        rgb_features = []
        for frame in rgb_frames:
            video = frame.split('/')[-4]
            utter = frame.split("/")[-3]
            index = int(frame.split("/")[-1].split(".")[0].split("_")[-1])/media/newssd/Aff-Wild_experiments/annotations
            path = os.path.join(self.pretrained_feature_root, video, utter+".mp4", "{:05d}.npy".format(index))
            try:
                rgb_features.append(np.load(path))
            except:
                raise ValueError("Incorrect feature path!")
        return [phase_batch_0,phase_batch_1, np.array(rgb_features)]
    def __len__(self):
        return len(self.utterance_list)

def imshow_grid(images, shape=[2, 9], name='default', save=False):
    """Plot images in a grid of a given shape."""
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        if i==17:
            break
        grid[i].axis('off')
        grid[i].imshow(images[i], cmap='gray', vmin=0., vmax=1.)  # The AxesGrid object work as a list of axes.

    plt.show() 
if __name__ == "__main__":
    data_dict = '/media/newssd/OMG_experiments/exps/test_dict.pkl'
    feature_root = '/media/newssd/OMG_experiments/Extracted_features/vgg_fer_features_fps=30_pool5'
    root_path = '/media/newssd/OMG_experiments/OMG_OpenFace_nomask/Test'
    label_name = 'arousal'
    train_dataset = Face_Dataset(root_path, feature_root, data_dict, label_name, py_level=4, py_nbands=2, sample_rate = 1, 
                                 num_phase=12, phase_size=48, test_mode=False, return_phase=False)
    #phase_batch0, phase_batch1, rgb_features, label, names = train_dataset[34]
    from Same_Length_Sampler import SameLengthBatchSampler
    train_sampler = SameLengthBatchSampler(train_dataset.indices_list, batch_size=4, drop_last=True, random=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
       batch_sampler= train_sampler, 
        num_workers=0, pin_memory=False )
    index= 0
    print([len(indices) for indices in train_dataset.indices_list])
    for data_batch in tqdm(train_loader):
        phase_batch0, phase_batch1, rgb_features, label, names = data_batch
        index+=1
        if index >=550:
            print()
    
    
