from pathlib import Path
import argparse
import yaml
import torch
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import sklearn
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix

from model.model import KMGCN
from dataloader import init_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def test(model, dataloader, path):
        # self.model.load_state_dict(torch.load('best_model/1_github/cn_ad/best_val_model_0.90625_128%.pth'))
        model.load_state_dict(torch.load(path))
        labels = []
        result = []
        pred = []

        model.eval()

        for data_in, pearson, pcorr, label, pseudo, ages, genders in dataloader:
            label = label.long()
            data_in, pearson, pcorr, label, pseudo, ages, genders = data_in.to(device), pearson.to(device), pcorr.to(device), label.to(device), pseudo.to(device), ages.to(device), genders.to(device)
            output = model(data_in, pearson, pcorr, pseudo, ages, genders)

            result += F.softmax(output, dim=1)[:, 1].tolist()
            pred += torch.argmax(F.softmax(output, dim=1), dim=1).tolist()
            labels += label.tolist()

        acc = accuracy_score(labels, pred)
        pre = precision_score(labels, pred)
        auc = sklearn.metrics.roc_auc_score(np.squeeze(labels), np.squeeze(pred))
        tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        f1 = f1_score(labels, pred)
        
        return acc, auc, pre, sen, spe, f1


def test1(model, path, datasets):
    train_dataset, val_dataset = datasets
    model.load_state_dict(torch.load(path))
    model.eval()

    # 存储相似度和对应索引的列表
    res = []
    count = 0
    for i in range(len(val_dataset)):
        data_in, pearson, pcorr, label, pseudo, age, gender = val_dataset[i]
        vector1 = pearson.flatten()

        data_ins = [data_in]
        pearsons = [pearson]
        pcorrs = [pcorr]
        labels = [label]
        pseudos = [pseudo]
        ages = [age]
        genders = [gender]

        similarity_list = []
        for j in range(len(train_dataset)):
            train_data_in, train_pearson, train_pcorr, train_label, train_pseudo, train_age, train_gender = train_dataset[j]
            vector2 = train_pearson.flatten()

            similarity = np.exp(-0.1*(np.linalg.norm(vector1.cpu().detach().numpy() - vector2.cpu().detach().numpy())))

            # 存储相似度和对应的索引
            similarity_list.append((similarity, j))

        # 按相似度降序排序
        sorted_similarity = sorted(similarity_list, key=lambda x: x[0], reverse=True)

        # 选取前15个相似度最大的样本
        top_15_indices = [index for similarity, index in sorted_similarity[:15]]
        # print('top_15_indices:',top_15_indices)
        for j in top_15_indices:
            train_data_in, train_pearson, train_pcorr, train_label, train_pseudo, train_age, train_gender = train_dataset[j]
            data_ins.append(train_data_in)
            pearsons.append(train_pearson)
            pcorrs.append(train_pcorr)
            labels.append(train_label)
            pseudos.append(train_pseudo)
            ages.append(train_age)
            genders.append(train_gender)

        # 获取对应的样本
        # top_15_samples = [self.train_dataset[index] for index in top_15_indices]    
        # print('top_15_samples:', len(top_15_samples))



        data_ins, pearsons, pcorrs, labels, pseudos, ages, genders = [np.stack(d) for d in (data_ins, pearsons, pcorrs, labels, pseudos, ages, genders)]
        data_ins, pearsons, pcorrs, labels, pseudos, ages, genders = [torch.from_numpy(d).float() for d in (data_ins, pearsons, pcorrs, labels, pseudos, ages, genders)]
        data_ins, pearsons, pcorrs, labels, pseudos, ages, genders = data_ins.to(device), pearsons.to(device), pcorrs.to(device), labels.to(device), pseudos.to(device), ages.to(device), genders.to(device)
        outputs  = model(data_ins, pearsons, pcorrs, pseudos, ages, genders)
        _, predicted = torch.max(outputs.data, 1)
        if labels[0] == predicted[0]:
            count += 1

    return count/len(val_dataset)

def main(args):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        print('config:',config)

        dataloaders, node_size, node_feature_size, timeseries_size, datasets = init_dataloader(config['data'])

        config['train']["seq_len"] = timeseries_size        
        config['train']["node_size"] = node_size            

        class_dim = 2 
        model = KMGCN(config, node_size, node_feature_size, timeseries_size, class_dim)
        model = model.to(device)

        path = 'best_model/best_val_model_0.84375_66%.pth'
        acc = test1(model, path, datasets)
        print('acc:', acc)
        


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/abide_PLSNet.yaml', type=str,
                        help='Configuration filename for training the model.')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    # 控制随机性
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    main(args)

