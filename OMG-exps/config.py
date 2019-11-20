import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of two stream emotion network")
parser.add_argument('--label_name', type=str, choices = ['arousal', 'valence', 'arousal_valence'])
parser.add_argument('--train_dict', type=str,default="../exps/train_dict.pkl") 
parser.add_argument('--val_dict', type=str, default="../exps/val_dict.pkl")
parser.add_argument('--test_dict', type=str, default = '../exps/test_dict.pkl')
parser.add_argument('--store_name', type=str, default="")
parser.add_argument("--save_root", type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument("--cnn", type =str, default = "resnet50", choices=[ 'resnet50', 'vgg'])
parser.add_argument('--root_path', type=str, default='/media/newssd/OMG_experiments/OMG_OpenFace')
parser.add_argument('--loss_type', type=str, default="ccc",
                    choices=['mse', 'ccc'])
parser.add_argument('--fusion', type=str, default= 'cat', choices =['cat','sum','product'])
parser.add_argument('--py_level', type=int, default=4) # including highpass and lowpass residual
parser.add_argument('--py_nbands', type=int, default=2) # number of orientations
parser.add_argument('--hidden_units', default=[2048, 256, 256, 1], type=int, nargs="+",
                    help='hidden units set up for MLP') # for spatial stream
parser.add_argument('--sample_rate', type=int, default = 1)
parser.add_argument('--length', type=int, default = 12)
parser.add_argument('--cat_before_gru', action='store_true')
parser.add_argument('--freeze', action='store_true')
# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--early_stop', type=int, default=3) # if validation loss didn't improve over 5 epochs, stop
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument( '--eval_batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--lr_steps', default=[3, 5], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: 20)')
parser.add_argument('--gradient_accumulation_steps', type=int, default = 1, 
                    help='accumulate gradient before loss backward.')
# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=25, type=int,
                    metavar='N', help='print frequency (default: 50) iteration')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 50) epochs')
# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='model')
parser.add_argument('--root_output',type=str, default='output')
parser.add_argument('--root_tensorboard', type=str, default='runs')
