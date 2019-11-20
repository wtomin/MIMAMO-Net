import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import numpy as np
class Pretrained_Model_Path(object):
    def __init__(self, label_name):
        self.label_name = label_name
    def get_mlp_path(self, five_fold=True):
        if self.label_name=='arousal':
            save_dir = '/media/newssd/OMG_experiments/two_layer_MLP_baseline_new_loss_5_fold/save_5_folds_results/label:arousal_cnn:resnet50_alpha:0_weight_decay:0.0005_momentum:0.9'
        elif self.label_name == 'valence':
            save_dir ='/media/newssd/OMG_experiments/two_layer_MLP_baseline_new_loss_5_fold/save_5_folds_results/label:valence_cnn:resnet50_alpha:0_weight_decay:0.001_momentum:0.9'
        if five_fold:
            return self.five_fold_model_path(save_dir)
        else:
            raise ValueError
    def get_phasenet_path(self, five_fold=True):
        if self.label_name =='arousal':
            save_dir = '/media/newssd/OMG_experiments/Phase_only_2DCNN_new_loss/sample_rate=1/phase_size=48/alpha=4/L=12/arousal_ccc_mse_py:4-2_loss_type:ccc_mse_gradient_accumulation_steps:1_batch_size:32_max_len:50_sample_rate:1_phase_size:48_alpha:4_L:12/model'
        elif self.label_name == 'valence':
            save_dir = '/media/newssd/OMG_experiments/Phase_only_2DCNN_new_loss/sample_rate=1/phase_size=48/alpha=4/L=12/valence_ccc_mse_py:4-2_loss_type:ccc_mse_gradient_accumulation_steps:1_batch_size:8_max_len:50_sample_rate:1_phase_size:48_alpha:4_L:12/model'
        if five_fold==True:
            return self.five_fold_model_path(save_dir)
        else:
            raise ValueError
        
    def five_fold_model_path(self, root_dir):
        model_paths = []
        for i in range(5):
            sub_folder = os.path.join(root_dir, 'fold_{}'.format(i))
            model_path = glob.glob(os.path.join(sub_folder, '*best_ccc*'))
            assert len(model_path)==1
            model_paths.append(model_path[0])
        return model_paths
class PhaseNet(nn.Module):
    def __init__(self, input_size, num_channels, hidden_size, output_dim , feature=False):
        super(PhaseNet,self).__init__()
        # input size : 2**i times 6 or 7
        if input_size not in [48, 96, 112]:
               raise ValueError("Incorrect input size")
        if input_size==48:
            num_conv_layers = 3
        else:
            num_conv_layers = 4
        if input_size==48 or input_size==96:
            last_conv_width = 6
        else:
            last_conv_width = 7
        self.conv_net = []
        for i in range(num_conv_layers):
            if i==0:
                self.conv_net.append(self._make_conv_layer(num_channels, 2**(i+6), kernel_size=3, stride=2))
            elif i==1:
                self.conv_net.append(self._make_conv_layer(num_channels+2**(i-1+6), 2**(i+6), kernel_size=3, stride=2))
            else:
                self.conv_net.append(self._make_conv_layer(2**(i-1+6), 2**(i+6), kernel_size=3, stride=2))
        last_conv_dim = 2**(i+6)
        self.conv_net = nn.ModuleList(self.conv_net)
        self.dropout = nn.Dropout2d(p=0.2)
        self.avgpool = nn.AvgPool2d(kernel_size=[last_conv_width, last_conv_width])
        fc4 = nn.Linear(last_conv_dim,hidden_size)
        fc5 = nn.Linear(hidden_size, output_dim)
        final_norm = nn.BatchNorm1d(1, eps=1e-6, momentum=0.1) # because 32/2400 =0.01
        self.mlp = nn.Sequential(fc4,
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.2),
                                 fc5,
                                 final_norm
                                 )
        self.feature = feature
    def _make_conv_layer(self, in_c, out_c, kernel_size = 3, stride = 2):
        ks = kernel_size 
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(ks, ks), padding=ks//2),
        nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(ks, ks), padding=ks//2,stride=stride),
        nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        )
        return conv_layer
    def forward(self, data_level0, data_level1):
        bs, num_frames,  num_channel, W0, H0 = data_level0.size()
        bs, num_frames, num_channel,  W1, H1 = data_level1.size()
        trans_data_level0 = data_level0.view(bs*num_frames, num_channel, W0, H0 )
        trans_data_level1 = data_level1.view(bs* num_frames, num_channel,  W1, H1)
        conv1 = self.conv_net[0](trans_data_level0)
        conv_out = torch.cat([conv1, trans_data_level1], dim=1) 
        for layer in self.conv_net[1:]:
            conv_out = self.dropout(layer(conv_out))
        avgpool = self.avgpool(conv_out)
        if self.feature:
            feature = avgpool.view(bs*num_frames, -1)
            for i, layer in enumerate(self.mlp):
                feature = layer(feature)
                if i==2:
                    return feature.view(bs, num_frames, -1)
        else:
            avgpool = avgpool.view(bs,num_frames, -1)
            avgpool_mean = avgpool.mean(1)
            out = self.mlp(avgpool_mean)
            return out
class MLP(nn.Module):
    def __init__(self, hidden_units, dropout=0.5, feature=False):
        super(MLP, self).__init__()
        self.feature = feature
        fc_list = []
        for i in range(len(hidden_units)-2):   
            fc_list += [nn.Linear(hidden_units[i], hidden_units[i+1]),
                       nn.ReLU(inplace=True),
                       nn.BatchNorm1d(hidden_units[i+1]),
                       nn.Dropout(dropout)]
        self.feature = feature
        self.fc = nn.Sequential(*fc_list) 
        self.classifier = nn.Sequential(nn.Linear(hidden_units[-2], hidden_units[-1]))
    def forward(self, input_tensor):
        out = self.fc(input_tensor)
        if not self.feature:
            out = self.classifier(out)
        return out
class  Two_Stream_RNN(nn.Module):
    def __init__(self, mlp_hidden_units, phase_size, phase_channels, phase_hidden_size, gru_hidden = 128, gru_num_layers=1, cat_before_gru=True, label_name='arousal', fusion='cat'):
        super(Two_Stream_RNN, self).__init__()
        self.mlp = MLP(mlp_hidden_units, feature=True)
        self.phase_net = PhaseNet(phase_size, phase_channels, phase_hidden_size, 1, feature=True)
        pretrained_path = Pretrained_Model_Path(label_name)
        mlp_paths =pretrained_path.get_mlp_path(five_fold=True)
        phasenet_paths = pretrained_path.get_phasenet_path(five_fold=True)
        self.mlp = self.load_model_weights(self.mlp, mlp_paths)
        self.phase_net = self.load_model_weights(self.phase_net, phasenet_paths)
        self.cat_before_gru = cat_before_gru
        self.fusion = fusion
        self.dropout = nn.Dropout(0.5)
        self.gru_hidden = gru_hidden
        self.gru_num_layers = gru_num_layers
        dp = 0.5 if self.gru_num_layers>1 else 0.
        if self.cat_before_gru:
            f_dim = 256
            self.rnns = nn.GRU(f_dim, self.gru_hidden, num_layers=self.gru_num_layers, bidirectional=True, dropout=dp)
            rnn_out_dim = self.gru_hidden
        else:
            f_dim = 256
            self.rnns_spatial = nn.GRU(f_dim, self.gru_hidden, num_layers=self.gru_num_layers, bidirectional=True, dropout=dp)
            self.rnns_temporal = nn.GRU(f_dim, self.gru_hidden, num_layers=self.gru_num_layers, bidirectional=True, dropout=dp)
            rnn_out_dim = self.gru_hidden*2
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(rnn_out_dim, 1),
                                        nn.BatchNorm1d(1))
    def load_model_weights(self, model, model_path):
        if isinstance(model_path, list):
            state_dict = {}
            state_dicts = []
            for m_p in model_path:
                ckp = torch.load(m_p)
                net_key = [key for key in ckp.keys() if key !='epoch'][0]
                m_state_dict = ckp[net_key]
                state_dicts.append(m_state_dict)
            for key in state_dicts[0].keys():
                try:
                    state_dict[key] = torch.mean(torch.stack([state_dicts[i][key] for i in range(len(model_path))], dim=0),dim=0)
                except:
                    #num_batches_tracked is int64 tensor, cannot use torch.mean
                    state_dict[key] = torch.mean(torch.stack([state_dicts[i][key].type('torch.FloatTensor') for i in range(len(model_path))], dim=0),dim=0)
                    state_dict[key] = state_dict[key].type('torch.LongTensor')
        else:
            ckp = torch.load(model_path)
            net_key = [key for key in ckp.keys() if (key !='epoch') and (key !='iter')][0]
            state_dict = ckp[net_key]
        model.load_state_dict(state_dict)
        return model
    def forward(self, data):
        phase_0 , phase_1, rgb_features = data
        bs, num_frames = phase_0.size(0), phase_0.size(1)
        rgb_features = rgb_features.view(bs* num_frames, -1)
        features_cnn = self.mlp(rgb_features)
        features_cnn = features_cnn.view(bs, num_frames,-1)
        features_phase = self.phase_net(phase_0, phase_1)
        if self.cat_before_gru:
           if self.fusion=='cat':
               features = torch.cat([features_cnn, features_phase], dim=-1)
            outputs_rnns = self.dropout(features)
            outputs_rnns, _ = self.rnns(outputs_rnns)
            outputs_rnns = F.relu(outputs_rnns)
            outputs_rnns = (outputs_rnns[:,:,:self.gru_hidden]+outputs_rnns[:,:,self.gru_hidden:])
        else:
            outputs_rnn_spatial = self.dropout(features_cnn)
            outputs_rnn_spatial, _ = self.rnns_spatial(outputs_rnn_spatial)
            outputs_rnn_spatial = F.relu(outputs_rnn_spatial)
            outputs_rnn_spatial = (outputs_rnn_spatial[:,:,:self.gru_hidden] + outputs_rnn_spatial[:,:,self.gru_hidden:])
            outputs_rnn_temporal = self.dropout(features_phase)
            outputs_rnn_temporal, _ = self.rnns_temporal(outputs_rnn_temporal)
            outputs_rnn_temporal = F.relu(outputs_rnn_temporal)
            outputs_rnn_temporal = (outputs_rnn_temporal[:,:,:self.gru_hidden] + outputs_rnn_temporal[:,:,self.gru_hidden:])
            if self.fusion=='cat':
                outputs_rnns = torch.cat([outputs_rnn_spatial, outputs_rnn_temporal], dim=-1)
        outputs_rnns = outputs_rnns[:,-1,:]
        out = self.classifier(outputs_rnns)
        return out

if __name__ == "__main__":
    root_path = '/media/newssd/OMG_experiments/OMG_OpenFace_nomask'
    feature_root = '/media/newssd/OMG_experiments/Extracted_features/resnet50_ferplus_features_fps=30_pool5_7x7_s1'
    label_name = 'arousal'
    from dataloader import Face_Dataset
    import os
    from Same_Length_Sampler import SameLengthBatchSampler
    L= 12
    batch_size = 16
    test_dataset = Face_Dataset(os.path.join(root_path,'Test'), feature_root, "../exps/test_dict.pkl", label_name, py_level=4, 
                               py_nbands=2,  sample_rate = 1, num_phase=L, phase_size=48, test_mode=True, return_phase=False)
    print("test dataset:{}".format(len(test_dataset)))
    test_batch_sampler =  SameLengthBatchSampler(test_dataset.indices_list, batch_size = batch_size, drop_last=False, random=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_sampler=test_batch_sampler,
        num_workers=0, pin_memory=False)
    model = Two_Stream_RNN([2048, 256, 256, 1], 48, 12*2, 256, cat_before_gru=False, gru_hidden = 128, gru_num_layers=1, fusion='cat')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params: {}".format(pytorch_total_params))
    model.cuda()
    
    for i, data in enumerate(test_loader):
        phase_0 , phase_1, rgb_features, labels, names = data
        phase_0 = phase_0.type('torch.FloatTensor').cuda()
        phase_1 = phase_1.type('torch.FloatTensor').cuda()
        rgb_features = rgb_features.type('torch.FloatTensor').cuda()
        out = model([phase_0, phase_1, rgb_features])
#    
    
                
