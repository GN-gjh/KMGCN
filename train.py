import os
import torch
import numpy as np
from datetime import datetime
from util import Logger, accuracy, TotalMeter

from pathlib import Path
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data
from util.loss import mixup_cluster_loss, GP_loss
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


class BasicTrain:
    def __init__(self, train_config, model, optimizers, dataloaders, log_folder, datasets) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader = dataloaders
        self.train_dataset, self.val_dataset = datasets
        self.epochs = train_config['epochs']
        self.pool_ratio = train_config['pool_ratio']
        self.optimizers = optimizers
        self.best_acc = 0
        self.best_acc_val = 0
        self.best_model = None
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.save_path = log_folder
        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy, self.edges_num = [
                TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def train_per_epoch(self, optimizer):
        self.model.train()
        for data_in, pearson, knowledge_graph, label, pseudo, ages, genders in self.train_dataloader:             
            label = label.long()
            data_in, pearson, knowledge_graph, label, pseudo, ages, genders = data_in.to(device), pearson.to(device), knowledge_graph.to(device), label.to(device), pseudo.to(device), ages.to(device), genders.to(device)
            inputs, nodes, knowledge_graph, targets_a, targets_b, lam = mixup_data(data_in, pearson, knowledge_graph, label, 1, device)      
            [output, score], data_graph, edge_variance = self.model(inputs, nodes, knowledge_graph, pseudo, ages, genders)           

            loss = 2 * mixup_criterion(self.loss_fn, output, targets_a, targets_b, lam)         
            loss += mixup_cluster_loss(data_graph, targets_a, targets_b, lam)   
            loss += 0.01*(1/F.mse_loss(knowledge_graph, data_graph))          
            loss += 0.001*GP_loss(score, self.pool_ratio)
            

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            self.edges_num.update_with_weight(edge_variance, label.shape[0])
    
    def val_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        pred = []

        self.model.eval()

        for data_in, pearson, knowledge_graph, label, pseudo, ages, genders in dataloader:
            label = label.long()
            data_in, pearson, knowledge_graph, label, pseudo, ages, genders = data_in.to(device), pearson.to(device), knowledge_graph.to(device), label.to(device), pseudo.to(device), ages.to(device), genders.to(device)
            [output, _], _, _ = self.model(data_in, pearson, knowledge_graph, pseudo, ages, genders)
            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            pred += torch.argmax(F.softmax(output, dim=1), dim=1).tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        acc = accuracy_score(labels, pred)
        pre = precision_score(labels, pred)
        tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        f1 = f1_score(labels, pred)

        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        con_matrix = confusion_matrix(labels, result)
        return [auc] + list(metric), con_matrix, acc, pre, sen, spe, f1

    def test(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        count = 0
        for i in range(len(self.val_dataset)):
            data_in, pearson, knowledge_graph, label, pseudo, age, gender = self.val_dataset[i]
            vector1 = pearson.flatten()
            data_ins = [data_in]
            pearsons = [pearson]
            knowledge_graphs = [knowledge_graph]
            labels = [label]
            pseudos = [pseudo]
            ages = [age]
            genders = [gender]

            similarity_list = []
            for j in range(len(self.train_dataset)):
                train_data_in, train_pearson, train_knowledge_graph, train_label, train_pseudo, train_age, train_gender = self.train_dataset[j]
                vector2 = train_pearson.flatten()
                similarity = np.exp(-0.1*(np.linalg.norm(vector1.cpu().detach().numpy() - vector2.cpu().detach().numpy())))
                similarity_list.append((similarity, j))
            sorted_similarity = sorted(similarity_list, key=lambda x: x[0], reverse=True)

            top_15_indices = [index for similarity, index in sorted_similarity[:15]]
            for j in top_15_indices:
                train_data_in, train_pearson, train_knowledge_graph, train_label, train_pseudo, train_age, train_gender = self.train_dataset[j]
                data_ins.append(train_data_in)
                pearsons.append(train_pearson)
                knowledge_graphs.append(train_knowledge_graph)
                labels.append(train_label)
                pseudos.append(train_pseudo)
                ages.append(train_age)
                genders.append(train_gender)

            data_ins, pearsons, knowledge_graphs, labels, pseudos, ages, genders = [np.stack(d) for d in (data_ins, pearsons, knowledge_graphs, labels, pseudos, ages, genders)]
            data_ins, pearsons, knowledge_graphs, labels, pseudos, ages, genders = [torch.from_numpy(d).float() for d in (data_ins, pearsons, knowledge_graphs, labels, pseudos, ages, genders)]
            data_ins, pearsons, knowledge_graphs, labels, pseudos, ages, genders = data_ins.to(device), pearsons.to(device), knowledge_graphs.to(device), labels.to(device), pseudos.to(device), ages.to(device), genders.to(device)
            [outputs, _], _, _ = self.model(data_ins, pearsons, knowledge_graphs, pseudos, ages, genders)
            _, predicted = torch.max(outputs.data, 1)
            if labels[0] == predicted[0]:
                count += 1

        return count/len(self.val_dataset)

    def generate_save_data_graph(self):
        data_graphs = []

        labels = []

        for data_in, nodes, pcorr, label, pseudo, ages, genders in self.test_dataloader:
            label = label.long()
            data_in, nodes, pcorr, label, pseudo, ages, genders = data_in.to(device), nodes.to(device), pcorr.to(device), label.to(device), pseudo.to(device), ages.to(device), genders.to(device)
            _, data_graph, _ = self.model(data_in, nodes, pcorr, pseudo, ages, genders)

            data_graphs.append(data_graph.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"data_graph.npy", {'matrix': np.vstack(
            data_graphs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results, txt):

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)
        with open(self.save_path / "training_info.txt", 'a', encoding='utf-8') as f:
            f.write(txt)
        torch.save(self.best_model.state_dict(), self.save_path/f"model_{self.best_acc}%.pt")



    def train(self):
        training_process = []
        txt = ''
        i =  0
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            val_result, _, val_acc, val_pre, val_sen, val_spe, val_f1 = self.val_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            model_path = 'best_model/nc_ad'
            if val_acc > self.best_acc_val:
                self.best_acc_val = val_acc
                print('best_acc_val:', self.best_acc_val)
                self.best_model = self.model
                model_filename = f'best_val_model.pth'
                model_filepath = os.path.join(model_path, model_filename)
                torch.save(self.model.state_dict(), model_filepath)

            
            if val_acc > 0.80:
                model_filename = f'best_val_model_{val_acc}_{i}%.pth'
                model_filepath = os.path.join(model_path, model_filename)
                torch.save(self.model.state_dict(), model_filepath)
                i += 1


            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Acc:{self.train_accuracy.avg: .3f}%',
                f'Val Acc:{self.val_accuracy.avg: .3f}%',
                f'val acc:{val_acc:.4f}',
                f'pre:{val_pre:.4f}',
                f'sen:{val_sen:.4f}',
                f'spe:{val_spe:.4f}',
                f'f1:{val_f1:.4f}'
            ]))

            txt += f'Epoch[{epoch}/{self.epochs}] '+f'Train Loss:{self.train_loss.avg: .3f} '+f'Train Accuracy:{self.train_accuracy.avg: .3f}% '+f'Val Accuracy:{self.val_accuracy.avg: .3f}% '+f'Val acc:{val_acc:.3f} '+f'pre:{val_pre:.4f}'+f'sen:{val_sen:.4f}'+f'spe:{val_spe:.4f}'+f'f1:{val_pre:.4f}'+'\n'

            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result)
        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        if self.best_acc > self.best_acc_val:
            self.save_path = self.save_path/Path(f"{self.best_acc: .3f}%_{date_time}")
        else:
            self.save_path = self.save_path/Path(f"{self.best_acc_val: .3f}%_{date_time}")
        self.save_result(training_process, txt)

