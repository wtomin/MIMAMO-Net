import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from config import parser
args = parser.parse_args()
import pickle
from network import Two_Stream_RNN
from dataloader import Face_Dataset, UtteranceRecord
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable as Variable
import copy
from tqdm import tqdm
import glob
from Same_Length_Sampler import SameLengthBatchSampler
import pandas as pd
class My_loss(torch.nn.Module): 
    def __init__(self):
        super().__init__()   
    def forward(self, x, y): 
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return -ccc
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

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output, args.root_tensorboard]
    folders_util = ["%s/"%(args.save_root) +folder for folder in folders_util]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)
         
def main():
    root_path = args.root_path
    label_name = args.label_name
    if args.cnn == 'resnet50':
        feature_root = '/media/newssd/OMG_experiments/Extracted_features/resnet50_ferplus_features_fps=30_pool5_7x7_s1'
    elif args.cnn == 'vgg':
        feature_root = '/media/newssd/OMG_experiments/Extracted_features/vgg_fer_features_fps=30_pool5'
    if len(args.store_name)==0:
        args.store_name = '_'.join( [label_name, 
                                     'cnn:{}'.format(args.cnn),
                                     'loss_type:{}'.format(args.loss_type),
                                      'batch_size:{}'.format(args.batch_size), 
                                      'cat_before_gru:{}'.format(args.cat_before_gru),
                                      'freeze:{}'.format(args.freeze),
                                      'fusion:{}'.format(args.fusion)]) 
    if len(args.save_root)==0:
        setattr(args, 'save_root', args.store_name)
    else:
        setattr(args, 'save_root', os.path.join(args.save_root, args.store_name))
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
    L = args.length
    train_dict = pickle.load(open(args.train_dict, 'rb'))
    val_dict = pickle.load(open(args.val_dict, 'rb'))
    train_dict.update(val_dict)
    train_val_dict = copy.copy(train_dict)
    video_names = sorted(list(train_dict.keys()))
    np.random.seed(0)
    video_indexes = np.random.permutation(len(video_names))
    video_names = [video_names[i] for i in video_indexes] 
    if args.test:
        run_5_fold_prediction_on_test_set(feature_root) 
    for i in range(5):
        ###########################  Modify the classifier ###################       
        model = Two_Stream_RNN(mlp_hidden_units=args.hidden_units, phase_size=48, phase_channels=2*L, 
                               phase_hidden_size=256, cat_before_gru=args.cat_before_gru, gru_hidden = 64, gru_num_layers=2, fusion=args.fusion)
        ###########################  Modify the classifier ###################   
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Params: {}".format(pytorch_total_params))
        phasenet_param = sum(p.numel() for p in model.phase_net.parameters() if p.requires_grad)
        print("Temporal Stream params: {} ({:.2f})".format( phasenet_param, phasenet_param/float(pytorch_total_params)))
        mlp_param = sum(p.numel() for p in model.mlp.parameters() if p.requires_grad)
        print("Spatial Stream params: {} ({:.2f})".format( mlp_param, mlp_param/float(pytorch_total_params)))
        model.cuda()
        if args.cat_before_gru:
            params_dict = [{'params': model.rnns.parameters(), 'lr':args.lr}, 
                            {'params': model.classifier.parameters(), 'lr':args.lr}, 
                            {'params': model.fusion_module.parameters(), 'lr':args.lr}]
        else:
            params_dict = [{'params': model.rnns_spatial.parameters(), 'lr':args.lr}, 
                           {'params': model.rnns_temporal.parameters(), 'lr':args.lr}, 
                           {'params': model.classifier.parameters(), 'lr':args.lr},
                           {'params': model.fusion_module.parameters(), 'lr':args.lr}]
        if not args.freeze:
            params_dict += [{'params': model.mlp.parameters(), 'lr':args.lr},
                             {'params': model.phase_net.parameters(), 'lr':args.lr}]
        optimizer = torch.optim.SGD(params_dict, # do not set learn rate for mlp and 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay) 
        torch.cuda.empty_cache()
        cudnn.benchmark = True  
        length = len(video_names)//5
        # five fold cross validation
        val_video_names = video_names[i*length:(i+1)*length]
        if i==4:
            val_video_names = video_names[i*length:]
        train_video_names = [name for name in video_names if name not in val_video_names]
        train_video_names = video_names #   delete it later
        train_dict = {key:train_val_dict[key] for key in train_video_names}
        val_dict = {key:train_val_dict[key] for key in val_video_names}
        train_dataset = Face_Dataset([os.path.join(root_path,'Train'), os.path.join(root_path,'Validation')], feature_root, train_dict, label_name, py_level=args.py_level, 
                                     py_nbands=args.py_nbands, sample_rate = args.sample_rate, num_phase=L, phase_size=48, test_mode=False,
                                     return_phase=False)
        val_dataset = Face_Dataset([os.path.join(root_path,'Train'), os.path.join(root_path,'Validation')], feature_root, val_dict, label_name, py_level=args.py_level, 
                                   py_nbands=args.py_nbands,  sample_rate = args.sample_rate, num_phase=L, phase_size=48, test_mode=True,
                                  return_phase=False)
        train_batch_sampler = SameLengthBatchSampler(train_dataset.indices_list, batch_size=args.batch_size, drop_last=True)
        val_batch_sampler  = SameLengthBatchSampler(val_dataset.indices_list, batch_size = args.eval_batch_size, drop_last=True, random=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_sampler=train_batch_sampler,
            num_workers=args.workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_sampler=val_batch_sampler,
            num_workers=args.workers, pin_memory=False)
         
        print("train dataset:{}".format(len(train_dataset)))
        print("val dataset:{}".format(len(val_dataset)))
        log = open(os.path.join(args.save_root, args.root_log, 'fold_{}.txt'.format(i)), 'w')
        output = "\n Fold: {}\n".format(i)
        log.write(output)
        log.flush()
        best_loss = 1000
        best_ccc = -100
        val_accum_epochs = 0
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args.lr_steps)
            train_mean, train_std = train(train_loader, model, criterion, optimizer, epoch, log)
            log_train_mean_std = open(os.path.join(args.save_root, args.root_log, 'mean_std_{}.txt'.format(i)), 'w')
            log_train_mean_std.write("{} {}".format(train_mean, train_std))
            log_train_mean_std.flush()
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
    run_5_fold_prediction_on_test_set(feature_root)
def run_5_fold_prediction_on_test_set(feature_root):
    test_dataset = Face_Dataset(os.path.join(args.root_path,'Test'), feature_root, args.test_dict, args.label_name, py_level=args.py_level, 
                               py_nbands=args.py_nbands,  sample_rate = args.sample_rate, num_phase=args.num_phase, phase_size=48, test_mode=True,
                               return_phase=False)
    print("test dataset:{}".format(len(test_dataset)))
    test_batch_sampler =  SameLengthBatchSampler(test_dataset.indices_list, batch_size = args.eval_batch_size, drop_last=False, random=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_sampler=test_batch_sampler,
        num_workers=args.workers, pin_memory=False)
    for i in range(5):
        file = open(os.path.join(args.save_root, args.root_log, 'mean_std_{}.txt'.format(i)), 'r')
        string = file.readline()
        train_mean, train_std = string.split(" ")
        train_mean = float(train_mean)
        train_std = float(train_std)
        # resume
        model = Two_Stream_RNN(mlp_hidden_units=args.hidden_units, phase_size=48, phase_channels=2*args.num_phase, phase_hidden_size=256, cat_before_gru=args.cat_before_gru)
        model.cuda()
        saved_model_path = os.path.join(args.save_root, args.root_model, 'fold_{}_best_ccc.pth.tar'.format(i))
        checkpoint = torch.load(saved_model_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print(("=> loading checkpoint '{}' epoch:{}".format(saved_model_path, start_epoch)))
        preds, names = test(test_loader, model,train_mean=train_mean, train_std=train_std)
        df= pd.DataFrame()
        df['video'] = pd.Series([n.split(" ")[0] for n in names])
        df['utterance'] = pd.Series([n.split(" ")[1] for n in names])
        df[args.label_name] = pd.Series([v for v in preds])
        df.to_csv(os.path.join(args.save_root, args.root_log, 'test_predictions_{}.csv'.format(i)), index=False)
def train(dataloader, model, criterion, optimizer, epoch, log): 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    optimizer.zero_grad()
    model.train()
    targets = []
    for i, data_batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        phase_0, phase_1, rgb_features, label, names  = data_batch
        phase_0 = Variable(phase_0.type('torch.FloatTensor').cuda())
        phase_1 = Variable(phase_1.type('torch.FloatTensor').cuda())
        rgb_features = Variable(rgb_features.type('torch.FloatTensor').cuda())
        label_var = Variable(label.type('torch.FloatTensor').cuda())
        out = model([phase_0, phase_1, rgb_features])
        loss= criterion(out.squeeze(-1), label_var)
        loss.backward()
        optimizer.step() # We have accumulated enought gradients
        optimizer.zero_grad()
        targets.append(label_var.data.cpu().numpy() )
        # measure elapsed time
        batch_time.update(time.time() - end)
        losses.update(loss.item(), label_var.size(0))
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
        torch.cuda.empty_cache()
    targets = np.concatenate([array for array in targets], axis=0)
    train_mean, train_std = np.mean(targets), np.std(targets)
    return train_mean, train_std

def validate(dataloader, model, criterion, iter, log, train_mean=None, train_std=None): 
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
#    df = pd.DataFrame(columns = ['video','utterance',str(args.label_name)+'_target', str(args.label_name)+'_prediction'])
    end = time.time()
    targets, preds = [], []
    for i, data_batch in enumerate(dataloader):
        phase_0, phase_1, rgb_features, label, names  = data_batch
        if (torch.sum(torch.isnan(phase_0))>0) or (torch.sum(torch.isnan(phase_1))>0) or (torch.sum(torch.isnan(rgb_features))>0):
            print()
        with torch.no_grad():
            phase_0 = Variable(phase_0.type('torch.FloatTensor').cuda())
            phase_1 = Variable(phase_1.type('torch.FloatTensor').cuda())
            rgb_features = Variable(rgb_features.type('torch.FloatTensor').cuda())
            label_var = Variable(label.type('torch.FloatTensor').cuda())
        out = model([phase_0, phase_1, rgb_features])
        targets.append(label_var.data.cpu().numpy() )
        preds.append(out.squeeze(-1).data.cpu().numpy())
        loss = criterion(out.squeeze(-1), label_var)  
        losses.update(loss.item(), label_var.size(0))
#        if np.isnan(losses.avg):
#            print() # caused by batch size =1
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
    mse_func =  mean_squared_error
    ccc_score = ccc(targets, preds)[0]
    mse_loss = mse_func(targets, preds)
    if train_mean is None:
        output = ' Validation : [{0}][{1}], ccc: {ccc_score:.4f} , loss:{loss_mse:.4f}'.format( i, len(dataloader), 
                          ccc_score=ccc_score, loss_mse = loss)  
    else:
        ccc_corr = ccc(targets, correct(preds, train_mean, train_std))[0]
        output = ' Validation : [{0}][{1}], ccc: {ccc_score:.4f}({ccc_corr:.4f}) , mse:{loss_mse:.4f}({loss_mse_c:.4f})'.format( i, len(dataloader), 
                          ccc_score=ccc_score, ccc_corr=ccc_corr, loss_mse = mse_loss, loss_mse_c = mse_func(targets, correct(preds, train_mean, train_std)))  
        ccc_score = ccc_corr
    print(output)
    log.write(output + '\n')
    log.flush() 
    return loss, ccc_score
def test(dataloader, model, train_mean=None, train_std=None): 
    print("Testing...")
    # switch to evaluate mode
    model.eval()
    preds = []
    names = []
    for i, data_batch in tqdm(enumerate(dataloader)):
        phase_0, phase_1, rgb_features, label, name_batch  = data_batch
        with torch.no_grad():
            phase_0 = Variable(phase_0.type('torch.FloatTensor').cuda())
            phase_1 = Variable(phase_1.type('torch.FloatTensor').cuda())
            rgb_features = Variable(rgb_features.type('torch.FloatTensor').cuda())
        out = model([phase_0, phase_1, rgb_features])
        preds.append(out.squeeze(-1).data.cpu().numpy())
        names.append(name_batch)
    preds =  np.concatenate([array for array in preds], axis=0)
    names =  np.concatenate([array for array in names], axis=0)
    preds = correct(preds, train_mean, train_std)
    return preds, names
def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = decay 
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
def save_checkpoint(state, is_best_loss, is_best_ccc, filename='fold'):
    torch.save(state, '%s/%s/%s_checkpoint.pth.tar' % (args.save_root, args.root_model,filename))       
    if is_best_ccc:
        shutil.copyfile('%s/%s/%s_checkpoint.pth.tar' % (args.save_root, args.root_model, filename),
                        '%s/%s/%s_best_ccc.pth.tar' % (args.save_root, args.root_model, filename)) 
        print("checkpoint saved to",  '%s/%s/%s_best_ccc.pth.tar' % (args.save_root, args.root_model,filename)) 
        
        
if __name__ == "__main__":
    main()
