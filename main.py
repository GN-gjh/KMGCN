from pathlib import Path
import argparse
import yaml
import torch
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn

import shutil


from train import BasicTrain

from model.model import  KMGCN
from dataloader import init_dataloader

def get_files_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    return files

def main(args):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        print('config:',config)

        dataloaders, node_size, node_feature_size, timeseries_size, datasets = init_dataloader(config['data'])

        config['train']["seq_len"] = timeseries_size        
        config['train']["node_size"] = node_size            


        class_dim = 2  
        model = KMGCN(config, node_size, node_feature_size, timeseries_size, class_dim)
        use_train = BasicTrain


        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'])
        opts = (optimizer,)

        save_folder_name = Path(config['train']['log_folder'])/Path(config['model']['name'])/Path(
            f"{config['data']['dataset']}_{config['data']['atlas']}")

        train_process = use_train(config['train'], model, opts, dataloaders, save_folder_name, datasets)                # use_train = BasicTrain (self, train_config, model, optimizers, dataloaders, log_folder)

        train_process.train()
        print('best acc:', train_process.best_acc_val)


        # test
        accs = []
        directory_path = 'best_model/nc_ad'
        files = get_files_in_directory(directory_path)
        for path in files:
            acc = train_process.test(path)
            accs.append(acc)
        max_acc = max(accs)
        max_acc_index = accs.index(max_acc)
        max_acc_file = files[max_acc_index]
        best_model_path = f'best_model/nc_ad/best_model.pth'
        shutil.copy(max_acc_file, best_model_path)
        for file in files:
            os.remove(file)

        acc = train_process.test(best_model_path)
        print('test acc:/t', acc)
        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/abide_PLSNet.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=100, type=int)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    main(args)
