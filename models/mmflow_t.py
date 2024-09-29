import torch.nn as nn
import torch
import numpy as np
import os
import sys
# sys.path.append("..")

import torch.nn.functional as F
from utils.model_utils import *
from utils import *


class MMFlow_T(nn.Module):
    
    def __init__(self, args):
        
        super(MMFlow_T,self).__init__()
        
    
        self.npoints = args.num_points
        self.stat_thres = args.stat_thres
        # self.npoints = 128
        # self.stat_thres = 0.5
        
        ## multi-scale set feature abstraction 
        sa_radius = [0.05, 0.1, 0.2, 0.4]
        sa_nsamples = [4, 8, 16, 32]
        sa_mlps = [32, 32, 64]
        sa_mlp2s = [64, 64, 64]
        #sa_mlps = [64, 64, 128]
        #sa_mlp2s = [128, 128, 128]
        num_sas = len(sa_radius)
        self.mse_layer = MultiScaleEncoder(sa_radius, sa_nsamples, in_channel=3, \
                                         mlp = sa_mlps, mlp2 = sa_mlp2s)
        
        ## context feature abstraction (the same as mse_layer)
        self.cfe_layer = MultiScaleEncoder(sa_radius, sa_nsamples, in_channel=3, mlp=sa_mlps, mlp2=sa_mlp2s)
            
        ## feature correlation layer (cost volumn)
        fc_inch = num_sas*sa_mlp2s[-1]*2  
        fc_mlps = [fc_inch,fc_inch,fc_inch]
        self.fc_layer = FeatureCorrelator(8, in_channel=fc_inch*2+3, mlp=fc_mlps)
        
        ## multi-scale set feature abstraction 
        ep_radius = [0.05, 0.1, 0.2, 0.4]
        ep_nsamples = [4, 8, 16, 32]
        ep_inch = fc_inch * 2 + 3
        ep_mlps = [fc_inch, int(fc_inch/2), int(fc_inch/8)]
        ep_mlp2s = [int(fc_inch/8), int(fc_inch/8), int(fc_inch/8)]
        num_eps = len(ep_radius)
        self.mse_layer2 = MultiScaleEncoder(ep_radius, ep_nsamples, in_channel=ep_inch, \
                                         mlp = ep_mlps, mlp2 = ep_mlp2s)
        
        ## Gated recurrent unit (GRU, fewer parameters than LSTM)
        gru_ch = int(num_eps * ep_mlp2s[-1])
        #self.gru = nn.GRU(input_size=num_eps * ep_mlp2s[-1], hidden_size=num_eps * ep_mlp2s[-1], num_layers=1)
        self.gru = nn.GRU(input_size=gru_ch , hidden_size=gru_ch, num_layers=1)
        
        ## heads
        sf_inch = num_eps * ep_mlp2s[-1]*2
        sf_mlps = [int(sf_inch/2), int(sf_inch/4), int(sf_inch/8)]
        self.fp = FlowHead(in_channel=sf_inch, mlp=sf_mlps)
        
        self.att_1 = Attention(256, 128)
        self.att_2 = Attention(256, 128)
        self.att_c = Attention(256, 128)
        self.att_t = Attention(512, 256)
    
    def Backbone(self,pc1,pc2,feature1,feature2,gfeat_prev):
        
        '''
        pc1: B 3 N
        pc2: B 3 N
        feature1: B 3 N
        feature2: B 3 N
        
        '''
        B = pc1.size()[0]
        N = pc1.size()[2]

        ## extract multi-scale local features for each point
        pc1_features = self.mse_layer(pc1,feature1)
        pc2_features = self.mse_layer(pc2,feature2)
        
        ## extract context feature for pc1
        pc1_contexts = self.cfe_layer(pc1, feature1)
        
        ## global features for each set
        #gfeat_1 = torch.max(pc1_features,-1)[0].unsqueeze(2).expand(pc1_features.size()[0],pc1_features.size()[1],pc1.size()[2])
        #gfeat_2 = torch.max(pc2_features,-1)[0].unsqueeze(2).expand(pc2_features.size()[0],pc2_features.size()[1],pc2.size()[2])
        weight_1 = self.att_1(pc1_features).unsqueeze(1) #B 1 N
        gfeat_1 = torch.bmm(weight_1, pc1_features.transpose(1, 2)).transpose(1, 2).expand(pc1_features.size()[0], pc1_features.size()[1], pc1.size()[2])
        weight_2 = self.att_1(pc2_features).unsqueeze(1) #B 1 N
        gfeat_2 = torch.bmm(weight_2, pc2_features.transpose(1, 2)).transpose(1, 2).expand(pc2_features.size()[0], pc2_features.size()[1], pc2.size()[2])
        weight_c = self.att_c(pc1_contexts).unsqueeze(1) #B 1 N
        gfeat_c = torch.bmm(weight_c, pc1_contexts.transpose(1, 2)).transpose(1, 2).expand(pc1_contexts.size()[0], pc1_contexts.size()[1], pc1.size()[2])
        
        ## concat local and global features
        pc1_features = torch.cat((pc1_features, gfeat_1),dim=1)
        pc2_features = torch.cat((pc2_features, gfeat_2),dim=1)
        pc1_contexts = torch.cat((pc1_contexts, gfeat_c), dim=1)
        
        ## associate data from two sets 
        cor_features = self.fc_layer(pc1, pc2, pc1_features, pc2_features)
        
        ## generate embeddings
        embeddings = torch.cat((feature1, pc1_contexts, cor_features), dim=1)
        prop_features = self.mse_layer2(pc1,embeddings)
        weight = self.att_2(prop_features).unsqueeze(1) #B 1 N
        gfeat = torch.bmm(weight, prop_features.transpose(1, 2)).transpose(1, 2).squeeze(2) # B 256
        
        ## update gfeat with GRU
        if gfeat_prev is None:
            gfeat_prev = torch.zeros(gfeat.shape).cuda()
        else :
            gfeat_prev = gfeat_prev.unsqueeze(2).repeat(1, 1, N)
            gfeat_prev = torch.cat((prop_features, gfeat_prev), dim = 1)
            weight1 = self.att_t(gfeat_prev).unsqueeze(1)
            gfeat_prev = torch.bmm(weight1, prop_features.transpose(1, 2)).transpose(1, 2).squeeze(2)

        self.gru.flatten_parameters()
        gfeat_new = self.gru(gfeat.unsqueeze(0), gfeat_prev.unsqueeze(0))[0].squeeze(0)
        gfeat_new_expand = gfeat_new.unsqueeze(2).expand(prop_features.size()[0], prop_features.size()[1], pc1.size()[2])

        final_features = torch.cat((prop_features, gfeat_new_expand),dim=1)
        
        return final_features, gfeat_new
                 
    def forward(self, pc1, pc2, feature1, feature2, gfeat_prev):
        
        # extract backbone features 
        final_features, gfeat = self.Backbone(pc1,pc2,feature1,feature2, gfeat_prev)
        
        # predict initial scene flow and classfication map
        output = self.fp(final_features)
        pred_f = torch.clamp(output, -0.1, 0.1)
    
        return pred_f, gfeat
    
    
class Attention(nn.Module):
    def __init__(self, inch, ouch):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(inch, ouch, 1, bias=False),
                                                 nn.BatchNorm2d(ouch),
                                                 nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(ouch, 1, 1, bias=False),
                                                 nn.BatchNorm2d(1))
    def forward(self, output):
        output = output.unsqueeze(3) # B 256 N 1 
        score = self.conv1(output)   # B 128 N 1 
        attention = self.conv2(score).squeeze(1).squeeze(2) # B 1 N 1
        
        return F.softmax(attention, dim=1)

if __name__ == '__main__':
    data1 = torch.randn(1,3,128)
    data2 = torch.randn(1,3,128)
    data3 = torch.randn(1,3,128)
    data4 = torch.randn(1,3,128)
    m = MMFlow_T()
    m(data1,data2,data3,data4)