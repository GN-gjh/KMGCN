import numpy as np
import torch
import torch.utils.data as utils

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

 
def init_dataloader(dataset_config):
    data = np.load(dataset_config["time_seires"], allow_pickle=True).item()                             

    final_fc = data["timeseires"]                                                               
    final_pearson = data["corr"]                                                                
    final_pcorr = data["pcorr"]
    labels = data["label"]                                                                  
    ages = data["age"]
    genders = data["gender"]

    _, _, timeseries = final_fc.shape                         

    _, node_size, node_feature_size = final_pearson.shape      

    scaler = StandardScaler(mean=np.mean(final_fc), std=np.std(final_fc))
    
    final_fc = scaler.transform(final_fc)                      


    pseudo = []     # 伪
    for i in range(len(final_fc)):
        pseudo.append(np.diag(np.ones(final_pearson.shape[1])))    


    if 'cc200' in  dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 200, 200))            
    elif 'aal' in dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 116, 116))
    elif 'cc400' in dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 392, 392))
    else:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 111, 111))


    final_fc, final_pearson, final_pcorr, labels, pseudo_arr, ages, genders = [torch.from_numpy(d).float() for d in (final_fc, final_pearson, final_pcorr, labels, pseudo_arr, ages, genders)]      # 转为torch向量

    length = final_fc.shape[0]
    train_length = int(length*dataset_config["train_set"])      # 168

    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        final_pcorr,
        labels,
        pseudo_arr,
        ages,
        genders
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_length, length-train_length])

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=True)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=True)


    return (train_dataloader, val_dataloader), node_size, node_feature_size, timeseries, (train_dataset, val_dataset)
