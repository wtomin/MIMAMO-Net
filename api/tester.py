from sampler.snippet_sampler import Snippet_Sampler
from video_processor import Video_Processor
from resnet50_extractor import Resnet50_Extractor
from phase_difference_extractor import Phase_Difference_Extractor
from mimamo_net import Two_Stream_RNN
from torch.autograd import Variable as Variable
import os
import torch
import tqdm
import numpy as np
import pandas as pd
from steerable.utils import get_device
device = get_device()
class Tester(object):
    def __init__(self,
             # parameters for testing
                 model_path , 
                 batch_size,
                 workers = 0,
                 # parameters for Video_Processor
                 save_size=112, nomask=True, grey=False, quiet=True,
                 tracked_vid=False, noface_save=False,
                 OpenFace_exe = 'OpenFace/build/bin/FeatureExtraction',
                 # parameters for Resnet50_Extractor
                 benchmark_dir = 'pytorch-benchmarks',model_name= 'resnet50_ferplus_dag',
                 feature_layer = 'pool5_7x7_s1',
                 # parameters for Snippet_Sampler
                 num_phase=12, phase_size = 48, 
                 length=64, stride=64,
                 # parameters for Phase_Difference_Extractor
                 height=4, nbands=2, scale_factor=2, 
                 extract_level = [1,2]
                 ):
        self.batch_size = batch_size
        self.workers = workers
        self.num_phase = num_phase
        self.phase_size = phase_size
        self.length = length
        self.stride = stride
        self.video_processor = Video_Processor(save_size, nomask, grey, quiet,
                              tracked_vid, noface_save, OpenFace_exe)
        self.resnet50_extractor =  Resnet50_Extractor(benchmark_dir, model_name, feature_layer)
        self.phase_difference_extractor = Phase_Difference_Extractor(height, nbands, scale_factor, 
                                          extract_level, not quiet)
        self.model = Two_Stream_RNN()
        assert os.path.exists(model_path)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("load checkpoint from {}, epoch:{}".format(model_path, start_epoch))
        self.model.to(device)
        self.label_name = ['valence', 'arousal'] # the pretrained model output format
    def test(self, input_video):
        video_name = os.path.basename(input_video).split('.')[0]
        # first input video is processed using OpenFace
        opface_output_dir = os.path.join(os.path.dirname(input_video), 
                video_name+"_opface")
        self.video_processor.process(input_video, opface_output_dir)
        # then the cropped and aligned faces are fed to resnet50_extractor
        feature_dir = os.path.join(os.path.dirname(input_video), 
                video_name+"_pool5")
        self.resnet50_extractor.run(opface_output_dir, feature_dir, video_name = video_name)
        
        # sampling images
        dataset = Snippet_Sampler(video_name, opface_output_dir, feature_dir,
        	annot_dir = None, label_name = 'valence_arousal',
            test_mode = True, num_phase=self.num_phase, phase_size = self.phase_size, 
            length=self.length, stride=self.stride)
        data_loader = torch.utils.data.DataLoader(
                       dataset,
                       batch_size=self.batch_size,
                       num_workers=self.workers, pin_memory=False)
        results = self.test_on_dataloader(data_loader, self.model)
        return results

    def test_on_dataloader(self, dataloader, model, train_mean=None, train_std=None):
        model.eval()
        sample_names = []
        sample_preds = []
        sample_ranges = []
        for i, data_batch in enumerate(dataloader):
            phase_f, rgb_f, label, ranges, names = data_batch
            with torch.no_grad():
                phase_f = phase_f.type('torch.FloatTensor').to(device)
                phase_0, phase_1 = self.phase_diff_output(phase_f, self.phase_difference_extractor)
                rgb_f = Variable(rgb_f.type('torch.FloatTensor').to(device))
                phase_0 = Variable(phase_0.type('torch.FloatTensor').to(device))
                phase_1 = Variable(phase_1.type('torch.FloatTensor').to(device))
            
            output = model([phase_0,phase_1], rgb_f)
            sample_names.append(names)
            sample_ranges.append(ranges)
            sample_preds.append(output.cpu().data.numpy())
        sample_names = np.concatenate([arr for arr in sample_names], axis=0)
        sample_preds = np.concatenate([arr for arr in sample_preds], axis=0)
        n_sample, n_length, n_labels = sample_preds.shape
        if train_mean is not None and train_std is not None:
            # scale 
            trans_sample_preds = sample_preds.reshape(-1, n_labels)
            trans_sample_preds = np.array([correct(trans_sample_preds[:, i], train_mean[i], train_std[i]) for i  in range(n_labels)])
            sample_preds = trans_sample_preds.reshape(n_sample, n_length, n_labels)
        sample_ranges = np.concatenate([arr for arr in sample_ranges], axis=0)
        video_dict = {}
        for video in sample_names:
            mask = sample_names==video
            video_ranges = sample_ranges[mask]
            if video not in video_dict.keys():
                max_len = max([ranges[-1] for ranges in video_ranges])
                video_dict[video] = np.zeros((max_len, n_labels))
            video_preds = sample_preds[mask]
            # make sure the dataset returns full range of video frames
            min_f, max_f = 0, 0
            for rg, pred in zip(video_ranges, video_preds):
                start, end = rg
                video_dict[video][start:end, :] = pred
                min_f = min(min_f, start)
                max_f = max(max_f, end)
            assert (min_f==0) and (max_f == max_len)
        for video in video_dict.keys():
            video_dict[video] = pd.DataFrame(data=video_dict[video], columns=self.label_name)
        return video_dict  
    def phase_diff_output(self, phase_batch, steerable_pyramid):
        """
        extract the first level and the second level phase difference images 
        """
        sp = steerable_pyramid
        bs, num_frames, num_phases, W, H = phase_batch.size()
        
        coeff_batch = sp.build_pyramid(phase_batch.view(bs*num_frames, num_phases, W, H))
        assert isinstance(coeff_batch, list)
        phase_batch_0 = sp.extract(coeff_batch[0])
        N, n_ch, n_ph, W, H= phase_batch_0.size()
        phase_batch_0 = phase_batch_0.view(N, -1, W, H)
        phase_batch_0 = phase_batch_0.view(bs, num_frames, -1, W, H)
        phase_batch_1 = sp.extract(coeff_batch[1])
        N, n_ch, n_ph, W, H= phase_batch_1.size()
        phase_batch_1 = phase_batch_1.view(N, -1, W, H)
        phase_batch_1 = phase_batch_1.view(bs, num_frames, -1, W, H)
        return phase_batch_0, phase_batch_1
