import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import glob
from torch import nn as nn
import torch
import torchvision
from utils.data_utils import GroupRandomHorizontalFlip,GroupRandomCrop,GroupScale,Stack,ToTorchFormatTensor

class VideoRecord(object):
    def __init__(self, video, feature_dir, annot_dir, label_name, test_mode = False):
        self.video = video
        self.feature_dir = feature_dir
        self.annot_dir = annot_dir
        self.label_name = label_name
        if self.label_name is not None:
            self.label_name = [self.label_name] if '_' not in self.label_name else self.label_name.split("_")
        self.test_mode = test_mode
        self.path_label = self.get_path_label()
     
    def get_path_label(self):
        frames = glob.glob(os.path.join(self.feature_dir, '*.npy'))
        frames = sorted(frames, key  = lambda x: os.path.basename(x).split(".")[0])
        if len(frames)==0:
            raise ValueError("number of frames of video {} should not be zero.".format(self.video))
        if not self.test_mode:
            assert self.label_name is not None, 'label name should not be None when test mode is False'
            assert self.annot_dir is not None, 'annot_dir should not be None when test mode is False'
            if (any([not os.path.exists(file) for file in annot_file])):
                raise ValueError("Annotation file not found when test mode is False")
            annot_file = [os.path.join(self.annot_dir, ln+".txt") for ln in self.label_name]
            total_labels = []
            for file in annot_file:
                f = open(file, "r")
                corr_frames, labels = [], []
                for i, x in enumerate(f):
                    label = float(x)
                    corr_frame = os.path.join(self.feature_dir, '{0:05d}.npy'.format(i+1))
                    if os.path.exists(corr_frame):
                        corr_frames.append(corr_frame)
                        labels.append(label)
                    else:
                        # skip those frames and labels
                        continue
                f.close()
                total_labels.append(labels)
            assert len(corr_frames) == len(labels)
            total_labels = np.asarray(total_labels)
            total_labels = total_labels.transpose(1, 0)
            return [corr_frames, total_labels]
        else:
            N = 1
            if self.label_name is not None:
                N = len(self.label_name)
            return [frames, np.array([[-100] * N]*len(frames))]

class Snippet_Sampler(data.Dataset):
    def __init__(self, video_name, root_path, feature_path,
                 annot_dir=None, label_name=None, test_mode =True, 
                 num_phase=12, phase_size = 48, 
                 length=64, stride=64,
                 verbose=False):
        '''Snippet_Sampler: An image sampler to sample snippets (a snippet consists of one RGB image and N
        greyscale images). The main inputs to it are two directories, one is the OpenFace processed faces directory,
        another is the directory containing renset50 feature vectors. 
        Parameters:
            video_name: string
                The processed video name
            root_path: string
                The directory path where cropped and aligned faces are stored (optionally other feature files).
            feature_path, string,
                The directory containing all renset50 feature vectors. 
            test_mode: bool, default False
                If True, it means the image sampler will only sample images, not annotations. Then annot_dir,
                label_name will not be used. And the labels will be dummy outputs.
                If False, the image sampler will sample both images and annotations. Then annot_dir, label_name
                need to be specified.
            label_name: string, default None
                If test_mode is False, the label_name needs to be one of {'arousal', 'valence', 'arousal_valence'}
            num_phase: int, default 12
                number of phase difference images, input to the phase net of the mimamo net.
            phase_size: int, default 48
                phase image size, default is 48x48
            length: int, default 64
                The length of snippets returned.
            stride: int, default 32
                The stride taken when sampling sequence of snippets. If stride<length, it means 
                adjacent sequence will overlap with each other.

        '''
        self.video_name = video_name
        self.root_path = root_path
        self.feature_path = feature_path
        self.annot_dir = annot_dir
        
        self.label_name = label_name
        self.test_mode = test_mode
        self.length = length 
        self.stride = stride 
        self.num_phase = num_phase
        self.phase_size = phase_size
        self.verbose = verbose
        self.parse_video()

    def parse_video(self):
        
        self.video_record = VideoRecord(self.video_name, self.feature_path, 
            self.annot_dir, self.label_name, self.test_mode)
        frames, labels = self.video_record.path_label
        self.seq_ranges = list()
        start, end = 0, self.length
        while end < len(frames) and (start<len(frames)):
            self.seq_ranges.append([start, end]) 
            start +=self.stride
            end = start+self.length

        if self.seq_ranges[-1][1] < len(frames):
            start = len(frames) - self.length
            end = len(frames)
            self.seq_ranges.append([start, end])  
        if self.verbose:        
            print("videos {}, number of seqs:{}".format(self.video_name, len(self.seq_ranges)))

    def __len__(self):
        return len(self.seq_ranges)
    def __getitem__(self, index):
        
        seq_ranges = self.seq_ranges[index]
        start, end = seq_ranges
        frames, labels = self.video_record.path_label
        seq_frames, seq_labels = frames[start:end], labels[start:end]
        # sample rgb images (features)
        imgs = []
        for f in seq_frames:
            imgs.append(np.load(f))
        # sample phase images 
        # figure ids
        sample_f_ids = []
        for f_id in range(start, end):
            phase_ids = []
            for i in range(self.num_phase+1):
                step = i-self.num_phase//2
                id_0 = max(0,f_id + step)
                id_0 = min(id_0, len(frames)-1) 
                phase_ids.append(id_0)
            sample_f_ids.append(phase_ids)
        sample_frames = [[frames[id] for id in ids] for ids in sample_f_ids]
        # load greyscale images
        phase_images= []
        for frames in sample_frames:
            phase_img_list = []
            for frame in frames:
                f_index = int(os.path.basename(frame).split(".")[0])
                img_frame = os.path.join(self.root_path, self.video_record.video+"_aligned", 
                    'frame_det_00_{:06d}.bmp'.format(f_index))
                try:
                   img = Image.open(img_frame).convert('L')
                except:
                    raise ValueError("incorrect face path")    
                phase_img_list.append(img)
            phase_images.append(phase_img_list)
        if not self.test_mode:
            random_seed = np.random.randint(250)
            phase_transform = torchvision.transforms.Compose(
                                   [GroupRandomHorizontalFlip(seed=random_seed),
                                   GroupRandomCrop(size=int(self.phase_size*0.85), seed=random_seed),
                                   GroupScale(size=self.phase_size),
                                   Stack(),
                                   ToTorchFormatTensor()])
        else:
            phase_transform = torchvision.transforms.Compose([
                                   GroupScale(size=self.phase_size),
                                   Stack(),
                                   ToTorchFormatTensor()]) 
        flat_phase_images = []
        for sublist in phase_images:
            flat_phase_images.extend(sublist)
        flat_phase_images = phase_transform(flat_phase_images)
        phase_images = flat_phase_images.view(len(phase_images), self.num_phase+1, self.phase_size, self.phase_size)
        return phase_images, np.array(imgs), np.array(seq_labels), np.array([start, end]), self.video_record.video
    

if __name__ == '__main__':
    root_path = '/media/newssd/Aff-Wild_experiments/Aligned_Faces_train'
    feature_path = '/media/newssd/Aff-Wild_experiments/Extracted_Features/Aff_wild_train/resnet50_ferplus_features_fps=30_pool5_7x7_s1'
    annot_dir = '/media/newssd/Aff-Wild_experiments/annotations'
    video_names = os.listdir(feature_path)[:25]

    train_dataset = Face_Dataset(root_path, feature_path, annot_dir, video_names, label_name='arousal_valence',  num_phase=12 , phase_size=48, test_mode=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = 4, 
        num_workers=0, pin_memory=False )
    for phase_f, rgb_f, label, seq_range, video_names in train_loader:
        phase_0, phase_1 = phase_f
        