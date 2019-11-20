import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
class Path(object):
    def __init__(self, label_name, fold_id):
        self.label_name = label_name
        self.fold_id = fold_id
        self.phase_path = {}
        a_root = '/media/newssd/Aff-Wild_experiments/phase_diff_5_fold/valence_loss_type:ccc_batch_size:64_alpha:1.0/model'
        v_root = '/media/newssd/Aff-Wild_experiments/phase_diff_5_fold/arousal_loss_type:ccc_batch_size:256_alpha:1.0/model'
        self.root_dir = a_root if label_name=='arousal' else v_root
    def path(self):
        model_path = glob.glob(os.path.join(self.root_dir, '*best_ccc*'))
        model_path = [path  for path in model_path if 'fold_{}'.format(self.fold_id) in path]
        assert len(model_path)==1
        return model_path[0]
class MLP(nn.Module):
    def __init__(self, hidden_units, dropout=0.3):
        super(MLP, self).__init__()
        input_feature_dim = hidden_units[0]
        num_layers = len(hidden_units)-1
        assert num_layers>0
        assert hidden_units[-1]==256
        fc_list = []
        for hidden_dim in hidden_units[1:]:
            fc_list += [ nn.Dropout(dropout),
                        nn.Linear(input_feature_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True)
                        ]
            input_feature_dim = hidden_dim
        self.mlp = nn.Sequential(*fc_list)
    def forward(self, input_tensor):
        bs, num_frames, feature_dim = input_tensor.size()
        input_tensor = input_tensor.view(bs*num_frames, feature_dim)
        out = self.mlp(input_tensor)
        return out.view(bs, num_frames, -1)
class PhaseNet(nn.Module):
    def __init__(self, input_size, num_channels, hidden_units=[256, 256, 1] , dropout=0.3, feature=False):
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
        fc_list =[]
        fc_list += [nn.Linear(last_conv_dim, hidden_units[0]),
                       nn.ReLU(inplace=True),
                       nn.BatchNorm1d(hidden_units[1]),
                       nn.Dropout(dropout)]
        for i in range(0, len(hidden_units)-2):
            fc_list += [nn.Linear(hidden_units[i], hidden_units[i+1]),
                       nn.ReLU(inplace=True),
                       nn.BatchNorm1d(hidden_units[i+1]),
                       nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc_list)
        final_norm = nn.BatchNorm1d(1, eps=1e-6, momentum=0.1) 
        self.classifier = nn.Sequential(nn.Linear(hidden_units[-2], hidden_units[-1]),
                                 final_norm )
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
        avgpool = avgpool.view(bs*num_frames, -1)
        out = self.fc(avgpool)
        if self.feature:
            return out
        else:
            out = self.classifier(out)
            return out
class Two_Stream_RNN(nn.Module):
    def __init__(self, mlp_hidden_units=[2048, 256, 256], dropout=0.5, label_name = 'arousal', fold_id = 0, num_phase=12):
        super(Two_Stream_RNN, self).__init__()
        self.mlp = MLP(mlp_hidden_units)
        self.num_phase = num_phase
        self.phasenet = PhaseNet(48, 2*num_phase, hidden_units =[256, 256, 1], dropout=0.3, feature=True)
        if '_' not in label_name:
            pretrained_path = Path(label_name, fold_id)
            self.phasenet = self.load_model_weights(self.phasenet, pretrained_path.path())
        self.transform = nn.Sequential(nn.Linear(512, 256),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm1d(256),
                                      nn.Dropout(dropout))
        self.rnns = nn.GRU(256, 128, bidirectional=True, num_layers=2, dropout = 0.3)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(256, len(label_name.split("_"))),
                                        nn.BatchNorm1d(len(label_name.split("_"))))
    def load_model_weights(self, model, model_path):
        ckp = torch.load(model_path)
        net_key = [key for key in ckp.keys() if (key !='epoch') and (key !='iter')][0]
        state_dict = ckp[net_key]
        model.load_state_dict(state_dict)
        return model
    def forward(self, phase_data, rgb_data):
        bs, num_frames = rgb_data.size(0), rgb_data.size(1)
        features_cnn = self.mlp(rgb_data)
        features_spatial = features_cnn.view(bs, num_frames, -1)
        phase_0, phase_1 = phase_data
        features_temporal = self.phasenet(phase_0, phase_1)
        features_temporal = features_temporal.view(bs, num_frames, -1)
        features = torch.cat([features_spatial, features_temporal], dim=-1)
        features = self.transform(features.view(bs*num_frames, -1))
        features = features.view(bs, num_frames, -1)
        outputs_rnns,  _ = self.rnns(features)
        outputs_rnns = outputs_rnns.view(bs* num_frames, -1)
        out = self.classifier(outputs_rnns)
        out = out.view(bs, num_frames, -1)
        return out
    
if __name__ == "__main__":

    from dataloader import Face_Dataset
    import os
    root_path = '/media/newssd/Aff-Wild_experiments/Aligned_Faces_train'
    feature_path = '/media/newssd/Aff-Wild_experiments/Extracted_Features/Aff_wild_train/resnet50_ferplus_features_fps=30_pool5_7x7_s1'
    annot_dir = '/media/newssd/Aff-Wild_experiments/annotations'
    video_names = os.listdir(root_path)[:50]
    train_dataset = Face_Dataset(root_path, feature_path, annot_dir, video_names, label_name='arousal', test_mode=False, num_phase=12, length=64, stride=32)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = 2, 
        num_workers=0, pin_memory=False )
    model = Two_Stream_RNN(mlp_hidden_units=[2048, 256, 256], fold_id=0)
    model.cuda()
    model.train()
    for phase_f, rgb_f, labels, ranges, videos in train_loader:
        phase_0, phase_1 = phase_f
        phase_0 = phase_0.type('torch.FloatTensor').cuda()
        phase_1 = phase_1.type('torch.FloatTensor').cuda()
        rgb_f = rgb_f.type('torch.FloatTensor').cuda()
        labels = labels.type('torch.FloatTensor').cuda()
        out = model([phase_0,phase_1], rgb_f)
        
        
