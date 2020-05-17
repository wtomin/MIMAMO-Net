import os
import sys
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
from sampler.image_sampler import Image_Sampler
import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils.model_utils import load_model, compose_transforms
import numpy as np
from steerable.utils import get_device
device = get_device()
class Resnet50_Extractor(object):
    def __init__(self, benchmark_dir = 'pytorch-benchmarks',model_name= 'resnet50_ferplus_dag',
                 feature_layer = 'pool5_7x7_s1'):
        ''' Resnet50_Extractor: A feature extractor to extractor the final convolutional layer's 
         feature vector (2048 dimensional) and save those feature vectors to npy file in an 
         output directory.

        Parameters: 
            benchmark_dir: string, default 'pytorch-benchmarks'
                The pytorch-benchmarks is installed in benchmark_dir.
            model_name: string, default 'resnet50_ferplus_dag'
                The model name for resnet50 model.
            feature_layer: string, default is 'pool5_7x7_s1'
                The output feature layer for resnet50 model is the final convolutional layer named
                'pool5_7x7_s1'.
        '''
        self.benchmark_dir = os.path.abspath(benchmark_dir)
        self.model_name = model_name
        self.feature_layer = feature_layer

        assert os.path.exists(self.benchmark_dir), 'benchmark_dir must exits'
        # load resnet50 model
        model_dir = os.path.abspath(os.path.join(self.benchmark_dir, 'ferplus'))
        self.model = load_model(self.model_name, model_dir)
        self.model = self.model.to(device)
        self.model.eval()
        # load transformation function
        meta = self.model.meta
        self.transform = compose_transforms(meta, center_crop=True)
    def run(self, input_dir, output_dir, batch_size=64):
        '''        
        input_dir: string, 
            The input_dir should have one subdir containing all cropped and aligned face images for 
            a video (extracted by OpenFace). The input_dir should be named after the video name.
        output_dir: string
            All extracted feature vectors will be stored in output directory.
        '''
        assert os.path.exists(input_dir), 'input dir must exsit!'
        assert len(os.listdir(input_dir)) != 0, 'input dir must not be empty!'
        
        video_name = os.path.basename(input_dir)
        dataset = Image_Sampler(video_name, input_dir, test_mode = True, transform=self.transform)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle=False, drop_last=False,
            num_workers=8, pin_memory=False )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
        	print("output_dir {} already exists, feature extraction skipped.".format(output_dir))
        	return
        with torch.no_grad():
            for ims, target, img_path, video_name in tqdm(data_loader):
                ims = ims.to(device)
                output = self.get_vec(ims)
                for feature, path, video_n in zip(output, img_path, video_name):
                    des_path = os.path.join(output_dir, "%05d.npy"%self.get_frame_index(path))
                    np.save(des_path, feature)
        return 
    def get_vec( self,  image):
        bs = image.size(0)
        layer = self.model._modules.get(self.feature_layer)
        my_embedding= torch.zeros([bs, 2048, 1, 1])
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        h = layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()
        return F.relu(my_embedding.squeeze())

    def get_frame_index(self, frame_path):
        frame_name = frame_path.split('/')[-1]
        frame_num  = int(frame_name.split('.')[0].split('_')[-1])
        return frame_num
