#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:37:28 2019
main
@author: ddeng
"""

import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import pickle
import torch.optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from config import parser
args = parser.parse_args()
from dataloader import Face_Dataset, VideoRecord
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable as Variable
from network import Two_Stream_RNN
import pandas as pd
from tqdm import tqdm
class My_loss(torch.nn.Module): 
    def __init__(self):
        super().__init__()   
    def forward(self, x, y): 
        cccs = 0
        for i in range(x.size(-1)):
            x_i = x[::, i]
            y_i = y[::, i]
            if len(x_i.size())==2 or len(y_i.size())==2:
                x_i = x_i.contiguous()
                y_i = y_i.contiguous()
                x_i = x_i.view(-1)
                y_i = y_i.view(-1)
            vx = x_i - torch.mean(x_i)
            vy = y_i - torch.mean(y_i)
            rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
            x_m = torch.mean(x_i)
            y_m = torch.mean(y_i)
            x_s = torch.std(x_i)
            y_s = torch.std(y_i)
            ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
            cccs+=ccc
        return -cccs
class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
    def forward(self, x):
        bs = x.size(0)
        if bs==1:
            x = x.squeeze()
            x = x.unsqueeze(0)
        else:
            x = x.squeeze()
        return x
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    v_pred = y_pred - pred_mean
    v_true = y_true - true_mean
    
    rho =  np.sum(v_pred*v_true) / (np.sqrt(np.sum(v_pred**2)) * np.sqrt(np.sum(v_true**2)))
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)
    
    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)
    return ccc, rho 

def correct(pred_val_y, train_mean, train_std):
    try:
        val_std = np.std(pred_val_y)
        mean = np.mean(pred_val_y)
        pred_val_y = np.array(pred_val_y)
    except:
        val_std = torch.std(pred_val_y)
        mean = torch.mean(pred_val_y)
    pred_val_y = mean + (pred_val_y - mean) * train_std / val_std
    return pred_val_y
def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output, args.root_tensorboard]
    folders_util = ["%s/"%(args.save_root) +folder for folder in folders_util]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)
def train(dataloader, model, criterion, optimizer, epoch, log): 
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    optimizer.zero_grad()
    model.train()
    for i, data_batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        phase_f, rgb_f, label, ranges, videos = data_batch
        phase_0, phase_1 = phase_f
        rgb_f = Variable(rgb_f.type('torch.FloatTensor').cuda())
        phase_0 = Variable(phase_0.type('torch.FloatTensor').cuda())
        phase_1 = Variable(phase_1.type('torch.FloatTensor').cuda())
        label_var = Variable(label.type('torch.FloatTensor').cuda())
        out = model([phase_0,phase_1], rgb_f)
        loss= criterion(out, label_var)
        loss.backward()
        optimizer.step() # We have accumulated enought gradients
        optimizer.zero_grad()
        # measure elapsed time
        batch_time.update(time.time() - end)
        losses.update(loss.item(), rgb_f.size(0))
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.6f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    .format( epoch, i, len(dataloader), batch_time=batch_time,
                        data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()

def validate(dataloader, model, criterion, iter, log, train_mean=None, train_std=None): 
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
#    df = pd.DataFrame(columns = ['video','utterance',str(args.label_name)+'_target', str(args.label_name)+'_prediction'])
    end = time.time()
    targets, preds = [], []
    for i, data_batch in enumerate(dataloader):
        phase_f, rgb_f, label, ranges, videos = data_batch
        with torch.no_grad():
            phase_0, phase_1 = phase_f
            rgb_f = Variable(rgb_f.type('torch.FloatTensor').cuda())
            phase_0 = Variable(phase_0.type('torch.FloatTensor').cuda())
            phase_1 = Variable(phase_1.type('torch.FloatTensor').cuda())
            label_var = Variable(label.type('torch.FloatTensor').cuda())
        output = model([phase_0,phase_1], rgb_f)
        targets.append(label_var.data.cpu().numpy() )
        preds.append(output.data.cpu().numpy())
        loss = criterion(output, label_var)  
        losses.update(loss.item(), rgb_f.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   i, len(dataloader), batch_time=batch_time, loss=losses))
            print(output)
            log.write(output + '\n')
            log.flush()
        torch.cuda.empty_cache()
    targets, preds = np.concatenate([array for array in targets], axis=0), np.concatenate([array for array in preds], axis=0)
    targets = targets.reshape(-1, targets.shape[-1])
    preds =  preds.reshape(-1, targets.shape[-1])
    mse_func =  mean_squared_error
    ccc_score = [ccc(targets[:, i], preds[:, i])[0] for i in range(targets.shape[-1])]
    mse_loss = [mse_func(targets[:, i], preds[:, i]) for i in range(targets.shape[-1])]
    ccc_corr =  [ccc(targets[:, i], correct(preds[:, i], train_mean[i], train_std[i]))[0]  for i in range(targets.shape[-1])]
    mse_corr = [mse_func(targets[:, i], correct(preds[:, i], train_mean[i], train_std[i])) for i in range(targets.shape[-1])]
    labels = args.label_name.split("_")
    list0 = ['ccc_{}: {:.4f}({:.4f}),'.format(labels[i], ccc_score[i], ccc_corr[i]) for i in range(targets.shape[-1])]
    list1 = ['mse_{}: {:.4f}({:.4f}),'.format(labels[i], mse_loss[i], mse_corr[i]) for i in range(targets.shape[-1])]
    output = ' '.join(['Validation : [{0}][{1}],'.format( i, len(dataloader)),
                        *list0, 
                        *list1])  
    ccc_score = ccc_corr
    print(output)
    log.write(output + '\n')
    log.flush() 
    return np.mean(mse_corr), np.mean(ccc_corr)
def test(dataloader, model, train_mean, train_std):
    model.eval()
    sample_names = []
    sample_preds = []
    sample_ranges = []
    for i, data_batch in tqdm(enumerate(dataloader)):
        phase_f, rgb_f, label, ranges, names = data_batch
        with torch.no_grad():
            phase_0, phase_1 = phase_f
            rgb_f = Variable(rgb_f.type('torch.FloatTensor').cuda())
            phase_0 = Variable(phase_0.type('torch.FloatTensor').cuda())
            phase_1 = Variable(phase_1.type('torch.FloatTensor').cuda())
        output = model([phase_0,phase_1], rgb_f)
        sample_names.append(names)
        sample_ranges.append(ranges)
        sample_preds.append(output.cpu().data.numpy())
    sample_names = np.concatenate([arr for arr in sample_names], axis=0)
    sample_preds = np.concatenate([arr for arr in sample_preds], axis=0)
    n_sample, n_length, n_labels = sample_preds.shape
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
    return video_dict        
def main():
    if args.cnn == 'vgg':
        feature_path = '/media/newssd/Aff-Wild_experiments/Extracted_Features/Aff_wild_train/vgg_vd_face_fer_dag_features_fps=30_fc7'
    elif args.cnn =='resnet50':
        feature_path = '/media/newssd/Aff-Wild_experiments/Extracted_Features/Aff_wild_train/resnet50_ferplus_features_fps=30_pool5_7x7_s1'
    label_name = args.label_name
    if len(args.store_name)==0:
        args.store_name = '_'.join( [label_name,
                                     'loss_type:{}'.format(args.loss_type),
                                      'batch_size:{}'.format(args.batch_size), 
                                      'length:{}'.format(args.length),
                                      'cnn:{}'.format(args.cnn),
                                      'mlp:{}'.format(args.hidden_units),
                                      'L:{}'.format(args.L)]) 
    setattr(args, 'save_root', args.store_name)
    print("save experiment to :{}".format(args.save_root))
    check_rootfolders()
    num_class = 1 if not "_" in args.label_name else 2
    setattr(args, 'num_class', num_class)
    if args.loss_type == 'mse':
        criterion = nn.MSELoss().cuda()
    elif args.loss_type=='ccc':
        criterion = My_loss().cuda()
    else: # another loss is mse or mae
        raise ValueError("Unknown loss type")           
    video_names = os.listdir(feature_path)
    # predict on test set. use the five fold models, and their average prediction as final results
    if args.test:
        test_lost_frames = '/media/newssd/Aff-Wild_experiments/test_set_lost_frames.pkl'
        test_lost_frames = pickle.load(open(test_lost_frames, 'rb'))
        if args.cnn =='resnet50':
            feature_path_test = '/media/newssd/Aff-Wild_experiments/Extracted_Features/Aff_wild_test/resnet50_ferplus_features_fps=30_pool5_7x7_s1'
        test_video_names = os.listdir(feature_path_test)
        test_root_path = '/media/newssd/Aff-Wild_experiments/Aligned_Faces_test'
        test_dataset = Face_Dataset(test_root_path, feature_path_test, '', test_video_names, label_name, test_mode=True, length=args.length, stride  = args.length, num_phase=args.L)
        test_loader = torch.utils.data.DataLoader(
                       test_dataset,
                       batch_size=args.batch_size//2,
                       num_workers=args.workers, pin_memory=False)
        for i in range(5):
            length = len(video_names)//5
            # five fold cross validation
            val_video_names = video_names[i*length:(i+1)*length]
            if i==4:
                val_video_names = video_names[i*length:]
            train_video_names = [name for name in video_names if name not in val_video_names]
            train_dataset = Face_Dataset(args.root_path, feature_path,  args.annot_dir, train_video_names, label_name, test_mode=False, length=args.length, stride=args.length)
            train_labels = train_dataset.total_labels
            train_mean, train_std = np.mean(train_labels, axis=0), np.std(train_labels, axis=0)
            model_path = os.path.join(args.save_root, args.root_model, 'fold_{}_best_ccc.pth.tar'.format(i))
            assert os.path.exists(model_path)
            model = Two_Stream_RNN(args.hidden_units, fold_id=i, num_phase=args.L)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            print("load checkpoint from {}, epoch:{}".format(model_path, start_epoch))
            model.cuda()
            video_dict = test(test_loader, model, train_mean, train_std)
            for video_name in video_dict:
                save_dir = os.path.join(args.save_root, args.root_log, 'fold_{}'.format(i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, '{}_{}.csv'.format(label_name, video_name))
                predictions = video_dict[video_name]
                test_frames = test_lost_frames[video_name]
                if test_frames['length'] == len(predictions):
                    # that means the prediction has the same length as original frames
                    pass
                elif test_frames['length'] > len(predictions):
                    # some frames are lost
                    success_frames = test_frames['success']
                    assert len(success_frames) == len(predictions)
                    new_predictions = np.ones(test_frames['length'])*(-100)
                    prev_pred = 0.
                    id = 0
                    for j in range(test_frames['length']):
                        if j+1 not in success_frames:
                            new_predictions[j] = prev_pred
                        else:
                            new_predictions[j] = predictions[id]
                            prev_pred = predictions[id]
                            id+=1 
                    predictions = new_predictions
                    assert id == len(success_frames)
                elif test_frames['length'] < len(predictions):
                    raise ValueError("prediction length incorrect!")
                assert predictions.shape[0] == test_frames['length']
                df = pd.DataFrame(predictions)
                df.to_csv(save_path, index=False, header=None)
        return
                
    for i in range(5):
    
        ###########################  Modify the classifier ###################       
        model = Two_Stream_RNN(args.hidden_units, fold_id=i, label_name=args.label_name, num_phase=args.L)
        ###########################  Modify the classifier ###################     
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Params: {}".format(pytorch_total_params))
        model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)  
        
        length = len(video_names)//5
        # five fold cross validation
        val_video_names = video_names[i*length:(i+1)*length]
        if i==4:
            val_video_names = video_names[i*length:]
        train_video_names = [name for name in video_names if name not in val_video_names]
        train_dataset = Face_Dataset(args.root_path, feature_path, args.annot_dir, train_video_names, label_name, test_mode=False, length=args.length, stride=args.length//2, num_phase=args.L)
        train_labels = train_dataset.total_labels
        train_mean, train_std = np.mean(train_labels, axis=0), np.std(train_labels, axis=0)
        val_dataset = Face_Dataset(args.root_path, feature_path, args.annot_dir, val_video_names, label_name, test_mode=False, length=args.length, stride=args.length//2, num_phase=args.L)
        train_loader =  torch.utils.data.DataLoader(
                        train_dataset, shuffle=True, 
                        batch_size = args.batch_size, drop_last=True,
                        num_workers=args.workers, pin_memory=False )
        val_loader = torch.utils.data.DataLoader(
                       val_dataset,
                       batch_size=args.batch_size//2,
                       num_workers=args.workers, pin_memory=False, drop_last=True)
        log = open(os.path.join(args.save_root, args.root_log, 'fold_{}.txt'.format(i)), 'w')
        output = "\n Fold: {}\n".format(i)
        log.write(output)
        log.flush()
        best_loss = 1000
        best_ccc = -100
        val_accum_epochs = 0
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args.lr_steps)
            train(train_loader, model, criterion, optimizer, epoch, log)
            torch.cuda.empty_cache() 
            if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
                loss_val, ccc_current_val = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log, train_mean, train_std)
                is_best_loss = loss_val< best_loss
                best_loss = min(loss_val, best_loss)
                is_best_ccc = ccc_current_val >best_ccc
                best_ccc  = max(ccc_current_val , best_ccc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                }, is_best_loss, is_best_ccc, filename='fold_{}'.format(i))
                if not is_best_ccc:
                    val_accum_epochs+=1
                else:
                    val_accum_epochs=0
                if val_accum_epochs>=args.early_stop:
                    print("validation ccc did not improve over {} epochs, stop".format(args.early_stop))
                    break
    
def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every N epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = decay         
def save_checkpoint(state, is_best_loss, is_best_ccc,filename='fold'):
    torch.save(state, '%s/%s/%s_checkpoint.pth.tar' % (args.save_root, args.root_model, filename))
    if is_best_loss:
        shutil.copyfile('%s/%s/%s_checkpoint.pth.tar' % (args.save_root, args.root_model, filename),
                        '%s/%s/%s_best_loss.pth.tar' % (args.save_root, args.root_model, filename)) 
        print("checkpoint saved to",  '%s/%s/%s_best_loss.pth.tar' % (args.save_root, args.root_model, filename))           
    if is_best_ccc:
        shutil.copyfile('%s/%s/%s_checkpoint.pth.tar' % (args.save_root, args.root_model, filename),
                        '%s/%s/%s_best_ccc.pth.tar' % (args.save_root, args.root_model, filename)) 
        print("checkpoint saved to",  '%s/%s/%s_best_ccc.pth.tar' % (args.save_root, args.root_model, filename))        
        
if __name__=='__main__':
    main()
