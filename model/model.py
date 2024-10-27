import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


from model.Encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Population(nn.Module):
    def __init__(self, roi_num=200, node_input_dim=200):
        super(Population, self).__init__()

        inner_dim = roi_num

        self.roi_num = roi_num
        self.node_input_dim = node_input_dim

        self.fc_p = nn.Sequential(nn.Linear(in_features=roi_num, out_features=roi_num),
                                  nn.LeakyReLU(negative_slope=0.2))

        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)


        self.fcn = nn.Sequential(
            nn.Linear(int(8 * int(roi_num * 0.7)), 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

        self.weight = torch.nn.Parameter(torch.Tensor(1, 8))

        self.softmax = nn.Sigmoid()

    def forward(self, x, pseudo, ages, genders):
        self.bz = x.shape[0]    
        bz = x.shape[0] 
        pseudo = self.fc_p(pseudo)         
        x = x + pseudo             
        topk = 8

        a = 0.9
        beta = 2
        alpha = 0.1
        num_nodes = x.shape[0]
        dist = np.zeros((num_nodes, num_nodes))
        inds = []
        for i in range(num_nodes):
            vector1 = x[i].flatten()                                                
            for j in range(num_nodes):
                vector2 = x[j].flatten() 
                similarity = np.exp(-alpha*(np.linalg.norm(vector1.cpu().detach().numpy() - vector2.cpu().detach().numpy())))
                if abs(ages[i] - ages[j]) <= beta and genders[i] == genders[j]:
                    dist[i][j] = similarity*a
                else:
                    dist[i][j] = similarity*(1-a)
            ind = np.argpartition(dist[i, :], -topk)[-topk:]
            inds.append(ind)                                         


        adj = np.zeros((num_nodes, num_nodes))                   
        for i, pos in enumerate(inds):
            for j in pos:
                adj[i][j] = dist[i][j]
        

        adj = torch.tensor(adj).to(device)
        adj = adj.to(torch.float)



        x = x.reshape((bz, -1))                    
        x = adj @ x
        x = x.reshape((bz, self.roi_num, -1))                   


        x = self.gcn(x)                                         

        x = x.reshape((bz*self.roi_num, -1))                   
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))                  




        x = x.reshape((bz, -1))                    
        x = adj @ x
        x = x.reshape((bz, self.roi_num, -1))                   

        x = self.gcn1(x)                                       

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))



        x = x.reshape((bz, -1))                   
        x = adj @ x
        x = x.reshape((bz, self.roi_num, -1))                  

        x = self.gcn2(x)                                       

        x = self.bn3(x)

        return x

class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=200):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)      

        return m

class GCNPredictor(nn.Module):
    def __init__(self, node_input_dim=200, roi_num=200):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)


        self.fcn = nn.Sequential(
            nn.Linear(int(8 * int(roi_num * 0.7)), 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )
        self.norm = torch.nn.LayerNorm(normalized_shape=roi_num, elementwise_affine=True)
        self.weight = torch.nn.Parameter(torch.Tensor(1, 8))

        self.softmax = nn.Sigmoid()


    def forward(self, node_feature, knowledge_graph, data_graph):
        bz = data_graph.shape[0]     # batch

        h1 = torch.einsum('ijk,ijp->ijp', knowledge_graph, node_feature)
        h2 = torch.einsum('ijk,ijp->ijp', data_graph, node_feature)       
        # x = h1 + h2
        x = h1 * h2

        x = self.gcn(x)                                        

        x = x.reshape((bz*self.roi_num, -1))                    
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))                  

        h1 = torch.einsum('ijk,ijp->ijp', knowledge_graph, x)
        h2 = torch.einsum('ijk,ijp->ijp', data_graph, x)                   
        # x = h1 + h2
        x = h1 * h2

        x = self.gcn1(x)                                       

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        h1 = torch.einsum('ijk,ijp->ijp', knowledge_graph, x)
        h2 = torch.einsum('ijk,ijp->ijp', data_graph, x)                 
        # x = h1 + h2
        x = h1 * h2

        x = self.gcn2(x)                                       

        x = self.bn3(x)

        return x

class Individual(nn.Module):

    def __init__(self, model_config, roi_num=200, node_feature_dim=200, time_series=78):
        super().__init__()

        self.extract = Encoder(input_dim=time_series, num_head=4, embed_dim=model_config['embedding_size'])                     

        self.emb2graph = Embed2GraphByProduct(model_config['embedding_size'], roi_num=roi_num)                          

        self.predictor = GCNPredictor(node_feature_dim, roi_num=roi_num)

        self.fc_p = nn.Sequential(nn.Linear(in_features=roi_num, out_features=roi_num),
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, t, nodes, knowledge_graph, pseudo):         
        x = self.extract(t)                 
        m = F.softmax(x, dim=-1)                                                                            
        data_graph = self.emb2graph(m)               

        bz, _, _ = data_graph.shape                 

        edge_variance = torch.mean(torch.var(data_graph.reshape((bz, -1)), dim=1))           

        pseudo = self.fc_p(pseudo)         
        nodes = nodes + pseudo             

        return self.predictor(nodes, knowledge_graph, data_graph), data_graph, edge_variance

class KMGCN(nn.Module):
    def __init__(self, config, node_size, node_feature_size, timeseries_size, out_dim):
        super(KMGCN, self).__init__()
        self.lam_I = 1
        self.lam_P = 1

        self.model_I = Individual(config['model'], node_size, node_feature_size, timeseries_size)        # corr[0] corr[1] t[1]
        self.model_P = Population(node_size, node_feature_size)

        self.weight = torch.nn.Parameter(torch.Tensor(1, 8))
        self.fcn = nn.Sequential(
            nn.Linear(int(8 * int(node_size * 0.7)), 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, out_dim)
        )
        self.softmax = nn.Sigmoid()
        

    
    def forward(self, inputs, nodes, knowledge_graph, pseudo, ages, genders):
        feature_I, data_graph, edge_variance = self.model_I(inputs, nodes, knowledge_graph, pseudo)
        feature_P = self.model_P(nodes, pseudo, ages, genders)

        bz = nodes.shape[0]
        x = self.lam_I*feature_I + self.lam_I*feature_P            

        score = (x * self.weight).sum(dim=-1)                 
        score = self.softmax(score)                           


        _, idx = score.sort(dim=-1)
        _, rank = idx.sort(dim=-1)  

        l = int(x.shape[1] * 0.7)
        x_p = torch.empty(bz, l, 8)

        for i in range(x.shape[0]):
            x_p[i] = x[i, rank[i, :l], :]

        x = x_p.view(bz,-1).to(device)                        

        out = self.fcn(x)

        return [out, score], data_graph, edge_variance
