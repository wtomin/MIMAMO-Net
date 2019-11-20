import os
import numpy as np
def read_val_ccc(txt_file):
    with open(txt_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    val_lines = [x for x in content if 'Validation' in x]
    val_cccs_v = [x.split('ccc_valence:')[-1].split(',')[0] for x in val_lines]
    val_cccs_v = [float(x.split('(')[-1].split(')')[0]) for x in val_cccs_v]
    val_cccs_a = [x.split('ccc_arousal:')[-1].split(',')[0] for x in val_lines]
    val_cccs_a = [float(x.split('(')[-1].split(')')[0]) for x in val_cccs_a]
    return [max(val_cccs_a), max(val_cccs_v)]
dirs = os.listdir('.')
for dir_path in dirs:
    if dir_path.startswith('arousal') or dir_path.startswith('valence'):
        fold_dir = os.path.join(dir_path, 'log')
        fold_txts = [os.path.join(fold_dir, 'fold_{}.txt'.format(i)) for i in range(5)]
        if all([os.path.exists(path) for path in fold_txts]):
            cccs = np.array([read_val_ccc(txt) for txt in fold_txts])
            print("name:{}".format(dir_path))
            print("5 fold arousal ccc {}, average ccc:{}".format(cccs[:, 0], np.mean(cccs[:, 0], axis=0)))
            print("5 fold valence ccc {}, average ccc:{}".format(cccs[:, 1], np.mean(cccs[:, 1], axis=0)))

