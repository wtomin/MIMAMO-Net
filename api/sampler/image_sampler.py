from PIL import Image, ImageOps
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import glob
import os
class VideoRecord(object):
    def __init__(self, video, face_dir, annot_dir, label_name, test_mode = False):
        '''        
        A VideoRecord that records all information of this video (and frames)
        '''
        self.video = video
        self.face_dir = face_dir
        self.annot_dir = annot_dir
        self.label_name = label_name 
        if self.label_name is not None:
            self.label_name = [self.label_name] if '_' not in self.label_name else self.label_name.split("_")
        self.test_mode = test_mode
        self.path_label = self.get_path_label()
     
    def get_path_label(self):
        '''        
        return all the frames and labels of this video
        '''
        
        frames = glob.glob(os.path.join(self.face_dir, self.video+"_aligned", '*.bmp'))
        frames = sorted(frames, key  = lambda x: os.path.basename(x).split(".")[0].split("_")[-1])
        if len(frames)==0:
            raise ValueError("number of frames of video {} should not be zero.".format(self.video))
        if not self.test_mode:
            assert self.label_name is not None, 'label name should not be None when test mode is False'
            assert self.annot_dir is not None, 'annot_dir should not be None when test mode is False'
            total_labels = []
            for label_name in self.label_name:
                annot_file = os.path.join(self.annot_dir, label_name+".txt")
                if (not os.path.exists(annot_file)):
                    raise ValueError("Annotation file not found.")
                else:
                    f = open(annot_file, "r")
                    corr_frames, labels = [], []
                    for i, x in enumerate(f):
                        label = float(x)
                        corr_frame = os.path.join(self.face_dir, self.video+"_aligned", 'frame_det_00_{0:06d}.bmp'.format(i+1))
                        if os.path.exists(corr_frame):
                            corr_frames.append(corr_frame)
                            labels.append(label)
                        else:
                            # skip those frames and labels
                            continue
                    f.close()
                    assert len(corr_frames) == len(labels)
                    total_labels.append(np.array(labels))
                total_labels = np.array(total_labels)
                assert len(total_labels) == len(corr_frames)
                return [corr_frames, total_labels]
        else:
            N = 1
            if self.label_name is not None:
                N = len(self.label_name)
             
            return [frames, np.array([[-100]*N]*len(frames))]

class Image_Sampler(data.Dataset):
    def __init__(self, video_name, root_path, test_mode =False,  
        annot_dir=None, label_name=None, transform = None, verbose=False,
        size = 224):
        ''' Image sampler for processed video (cropped & aligned faces), which generates an iterator, 
        each time returning a tuple containing (an image (torch.tensor), label, frame path, video name)

    Parameters: 
        video_name: string
            The processed video name
        root_path: string
            The directory path where cropped and aligned faces are stored (optionally other feature files).
        test_mode: bool, default False
            If True, it means the image sampler will only sample images, not annotations. Then annot_dir,
            label_name will not be used. And the labels will be dummy outputs.
            If False, the image sampler will sample both images and annotations. Then annot_dir, label_name
            need to be specified.
        annot_dir: string, default None
            If test_mode is False, the annot_dir needs to be a directory containing arousal.txt or valence.txt
            or both.
        label_name: string, default None
            If test_mode is False, the label_name needs to be one of {'arousal', 'valence', 'arousal_valence'}
        transform: torchvision.transforms.Compose object, default is None
            Transformation functions for images.
        verbose: bool, default False
            Whether to print out video information.
        size: int, default 112
            sampled image size.
        '''
        self.video_name = video_name
        self.root_path = root_path
        self.annot_dir = annot_dir
        self.label_name = label_name
        self.test_mode = test_mode
        self.transform = transform
        self.size = size
        self.verbose = verbose
        self.parse_video()
        if self.transform is None:
            self._create_transform()
        assert self.transform is not None
        
    def parse_video(self):
        self.video_record = VideoRecord(self.video_name, self.root_path, 
            self.annot_dir, self.label_name, self.test_mode)
        frames, labels = self.video_record.path_label
        if self.verbose:
            print("video {} has {} frames".format(self.video_name, len(frames)))
        self.frame_ids = np.arange(len(frames))
    def __len__(self):
        return len(self.frame_ids)
    def __getitem__(self, index):
        f_id = self.frame_ids[index]
        frames, labels = self.video_record.path_label
        frame, label = frames[f_id], labels[f_id]
        img = Image.open(frame)
        img = self.transform(img)
        return img, label, frame, self.video_record.video

    def _create_transform(self):
        if not self.test_mode:
            img_size = self.size
            resize = int(img_size * 1.2)
            transform_list = [transforms.Resize(resize),
                              transforms.RandomCrop(img_size),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),
                            ]
        else:
            img_size = self.size
            transform_list = [transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                            ]
        self.transform = transforms.Compose(transform_list)