#2023.12 for Geoexplainer: Interpreting graph convolutional networks with geometric masking
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import time
import os
import numpy as np
from numpy import *
from .processor import Processor
from math import sqrt
from torch.nn.functional import cross_entropy
from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new, plucker

from torch_geometric.nn import MessagePassing
import tools.utils as utils

import pandas as pd
import random
import torchsnooper
import scipy
from scipy import stats
import scipy.stats as stat
from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p
import csv
import cv2
import sklearn
from sklearn.decomposition import PCA
from scipy.interpolate import make_interp_spline
from scipy.integrate import simps
from pgmpy.estimators.CITests import chi_square
import torch.nn.functional as F

max_body = 2
num_joint = 25
max_frame = 300

coeffs = {
    'mask_size': 0.005,
    'mask_ent': 1.0,
}


def data_reshape(data, mask):
    N, C, T, V, M = data.size()
    reshaped_mask = torch.reshape(mask, [N, T, V, M, C])
    reshaped_mask = reshaped_mask.permute(0, 4, 1, 2, 3).contiguous()

    return reshaped_mask


class Explainer(Processor):
    def __init__(self, argv=None):
        super().__init__(argv)
        self.epochs = 100

    def start(self):
        print("Using {} weights.".format(self.arg.valid))
        explainer_dict = {'work1': self.explainer}
        self.method = self.arg.exp_type.lower()
        out = self.process_exps(explainer_dict[self.method])

        return
    # @torchsnooper.snoop()
    def process_exps(self, exp_func):

        time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
        file_txt = self.file_txt = open(f'result-forscrew-20240325/{self.method}-ntu-{self.arg.valid}-{time_now}.txt', mode='a', encoding="utf-8")
        
        if self.arg.data_level == 'instance':
            label_name_path = './data/label_name.txt'
            with open(label_name_path) as f:
                label_name = f.readlines()
                label_name = [line.rstrip() for line in label_name]
                self.label_name = label_name
            return 

        if self.arg.data_level == 'test_set':
            count = 0
            all_un_screw_node_fidelity_prob_out = 0
            all_un_screw_node_fidelity_acc_out = 0
            Sparsity_out = 0
            
            loader = self.data_loader['test'] # Validation
            for data, label in loader:
                self.model.eval() 
                self.model.to(self.dev)
                data_torch = data 
                data_model = data.float().to(self.dev)

                with torch.no_grad():
                    out,_= self.model(data_model)    
                    label = out.argmax(dim=1, keepdim=True).item() 
                    probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
                N, C, T, V, M = data.size()
                x = data.permute(0, 2, 3, 4, 1).contiguous()
                x = x.view(N * T * V * M, C)
                frame_mask = data_model
                screw_all_mask = self._set_screw_mask_(data, label)
                node_feature_mask,_ = exp_func( screw_all_mask,frame_mask, data,data_torch, x, label)
                all_un_screw_node_fidelity_acc,all_un_screw_node_fidelity_prob,Sparsity\
                    = self.evaluate(frame_mask, screw_all_mask, node_feature_mask, data, x, label, probability)
                Sparsity_out += Sparsity
                all_un_screw_node_fidelity_acc_out += all_un_screw_node_fidelity_acc
                all_un_screw_node_fidelity_prob_out += all_un_screw_node_fidelity_prob
                count = count + 1
                print('count is',count)
                if count >= 2000:
                    break    

            Sparsity = Sparsity_out / count
            all_un_screw_node_fidelity_acc = all_un_screw_node_fidelity_acc_out / count 
            all_un_screw_node_fidelity_prob = all_un_screw_node_fidelity_prob_out / count 
            print("=================================================================")
            file_txt.write("=================================================================" + '\n')
            print("evaluate")
            file_txt.write("evaluate" + '\n')
            print("all_un_screw_node_fidelity_acc:", format(all_un_screw_node_fidelity_acc, '.3f'))
            file_txt.write("all_un_screw_node_fidelity_acc:" + str(format(all_un_screw_node_fidelity_acc, '.3f')) + '\n')            
            print("all_un_screw_node_fidelity_prob:", format(all_un_screw_node_fidelity_prob, '.3f'))
            file_txt.write("all_un_screw_node_fidelity_prob:" + str(format(all_un_screw_node_fidelity_prob, '.3f')) + '\n')           
            print("Sparsity:", format(Sparsity, '.3f'))
            file_txt.write("Sparsity:" + str(format(Sparsity, '.3f')) + '\n')
            file_txt.close()
            print(file_txt)
            torch.cuda.empty_cache()
            return

    def lead_data(self):
        skeleton_name = self.arg.skeleton
        action_class = int(
            skeleton_name[skeleton_name.find('A') + 1:skeleton_name.find('A') + 4]) - 1
        print("Processing skeleton:", skeleton_name)
        print("Skeleton class:", action_class)
        self.action_class = action_class
        skeleton_file_path = './data/ntuu/'
        skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
        data_numpy = plucker(
            skeleton_file, max_body=max_body, num_joint=num_joint)
        self.model.eval()
        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data_torch = torch.from_numpy(data_numpy)
        data_model = data.float().to(self.dev)
        with torch.no_grad():
            out,_= self.model(data_model)
            label = out.argmax(dim=1, keepdim=True)
            probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
        N, C, T, V, M = data.size()
        x = data.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(N * T * V * M, C)
        return data,data_torch, x, label, probability 

    def Antisymmetry(self,sdata):
        l,m,n = sdata[0],sdata[1],sdata[2]
        return np.array([[0,-n,m],[n,0,-l],[-m,l,0]])

    def _set_screw_mask_(self, data_torch, label):
        #==20220512==========================================
        theta = 180*np.pi/180 
        s = torch.randn(3)
        d = torch.randn(3)
        I = np.array([[1,0,0],[0,1,0],[0,0,1]])
        As = self.Antisymmetry(s)
        A = self.Antisymmetry(d)
        s1 = s.reshape(len(s),1)
        s2 = s.reshape(1,len(s))
        sst = np.dot(s1,s2)    
        R = cos(theta)*I + sin(theta)*As + (1-cos(theta))*sst
        AR = np.matmul(A,R)
        O = np.zeros([3,3])
        Ad = np.vstack((np.hstack((R,O)),np.hstack((AR,R))))
        std = 0.1
        data_dot=data_torch  
        data_torch=data_torch.squeeze(0)  
        c, t, v, m = data_torch.size()
        data_dot = data_torch.detach().cpu().numpy()
        data_torch = data_torch.detach().cpu().numpy()
        for k in range(v):
            L = data_torch[:,:,k,:]
            for i in range(t):
                for j in range(m):          
                    Lp = np.matmul(Ad,L[:,i,j])
                    data_dot[:,i,k,j] = Lp
        for o in range(c):
            for i in range(t):
                for j in range(m):
                    C=data_dot[o,i,:,j]
                    minVals = np.min(C)    
                    maxVals = np.max(C)
                    for k in range(v):
                        if data_dot[o,i,k,j]== 0:
                            data_dot[o,i,k,j] = 0
                        else:
                            normLp = (data_dot[o,i,k,j]-minVals)/(maxVals-minVals)
                            data_dot[o,i,k,j] = normLp
        normdata_dot = data_dot
        normdata_dot = torch.from_numpy(normdata_dot)
        normdata_dot = normdata_dot.unsqueeze(0)
        N, C, T, V, M = normdata_dot.size() 
        normdata_dot = normdata_dot.permute(0, 2, 3, 4, 1).contiguous()
        normdata_dot = normdata_dot.view(N * T * V * M, C)
        self.screw_all_mask = torch.nn.Parameter(normdata_dot * std)
        screw_all_mask = self.screw_all_mask
        return screw_all_mask

    def _set_feature_masks_(self, x):
        std = 0.5
        if self.arg.feat_mask_type == 'individual_feature':
            N, F = x.size()
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__node_feat_mask__ = self.node_feat_mask

    def clear_masks(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__node_feat_mask__ = None
        self.node_feat_masks = None

    def calculation(self, data, label):
        self.model.eval()
        data = data.float().to(self.dev)
        with torch.no_grad():
            out,_= self.model(data)
            probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
        return probability  

    def __loss__(self, masked_pred, original_pred, mask):    
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        l1_coeff = 0.01
        loss = cce_loss+ l1_coeff * torch.mean(torch.abs(1 - mask))
        return loss

    def forward(self, frame_mask, data, change_data, label):
        self.model.eval()
        self.clear_masks()
        self._set_frame_mask_(data, label)
        self._set_screw_mask_(data, label)
        frame_mask,mask = self.explainer(frame_mask, screw_all_mask, data, change_data, label)
        self.clear_masks()
        return mask
    

    def explainer(self, screw_all_mask, frame_mask, data, data_torch,change_data, label):      
        change_data = change_data.to(self.dev)
        self.model.zero_grad()
        self.model.eval()
        data = data.float().to(self.dev).requires_grad_()
        with torch.no_grad():
            output,_ = self.model(data)
            prediction_label = output.argmax(dim=1, keepdim=True)
        self._set_feature_masks_(change_data)
        node_feat_mask = self.node_feat_mask
        node_feat_mask = data_reshape(data=data, mask=node_feat_mask).float().to(self.dev)
        screw_all_mask = screw_all_mask.detach()
        screw_all_mask = data_reshape(data=data, mask=screw_all_mask).float().to(self.dev)
        node_feat_mask = torch.nn.Parameter(node_feat_mask.clone().detach().requires_grad_(True))
        screw_all_mask = torch.nn.Parameter(screw_all_mask.clone().detach().requires_grad_(True))
        parameters = [node_feat_mask,screw_all_mask]
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        bast_output = torch.tensor([0]).to(self.dev)
        index = 0
        for epoch in range(1, self.epochs + 1):
            self.model.eval()
            self.model.to(self.dev)
            h=node_feat_mask.sigmoid().to(self.dev)*screw_all_mask.to(self.dev)
            out,_ = self.model(data*h.float())
            test_out = torch.nn.functional.softmax(out[0], dim=-1).to(self.dev)
           
            if test_out[label] > bast_output:
                bast_output = test_out[label]
                index = epoch
            else:
                if epoch - index > 5:
                    break
           
            loss = self.__loss__(out, prediction_label[0], mask=h)           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        best_feat_mask = self.node_feat_mask.detach().sigmoid()
        bast_screw_mask = screw_all_mask.detach().sigmoid()
        self.clear_masks()
        
        return best_feat_mask,bast_screw_mask
    
    
    def GNN_Explainer(self):
        return

    def evaluate(self, frame_mask, screw_all_mask,node_feature_mask, data, x, label, probability):
        return self.fidelity(frame_mask ,screw_all_mask,node_feature_mask, data, x, label, probability)

    def fidelity(self, frame_mask,screw_all_mask, node_feature_mask, data, x, label, probability):
        file_txt = self.file_txt
        data = data.to(self.dev)
        frame_mask = frame_mask.to(self.dev)
        All_one_mask = torch.ones_like(data).to(self.dev)
        node_mask = data_reshape(data, node_feature_mask).to(self.dev)
        screw_all_mask = data_reshape(data, screw_all_mask).to(self.dev)
        un_frame_mask = (All_one_mask - frame_mask).to(self.dev)
        un_node_mask = (All_one_mask - node_mask).to(self.dev)
        un_screw_mask = (All_one_mask - screw_all_mask).to(self.dev)
        
        print("label：", label, "   probability：", format(probability, '.4f'))
        file_txt.write("label：" + str(label) + "   probability：" + str(format(probability, '.4f')) + '\n')
        self.model.to(self.dev)
        self.model.eval()
        with torch.no_grad():    
            all_un_screw_in = (data*un_screw_mask* un_node_mask).float().to(self.dev)
            all_un_screw_out,_ = self.model(all_un_screw_in)
            all_un_screw_label = all_un_screw_out.argmax(dim=1, keepdim=True).item()
            all_un_screw_prob = torch.nn.functional.softmax(all_un_screw_out[0], dim=-1)[label].item()
            print("all_un_screw_label：", all_un_screw_label, "     all_un_screw_label：",
                  format(all_un_screw_prob, '.4f'))
            file_txt.write("all_un_screw_label：" + str(all_un_screw_label) + "     all_un_screw_label：" + str(
                format(all_un_screw_prob, '.4f')) + '\n')


        acc_un_screw_node = 1 if label == all_un_screw_label else 0 
        prob_un_screw_node = probability - all_un_screw_prob 
        all_un_screw_node_fidelity_acc = 1 - acc_un_screw_node
        print("all_un_screw_node_fidelity_acc:", all_un_screw_node_fidelity_acc)
        file_txt.write("all_un_screw_node_fidelity_acc:" + str(all_un_screw_node_fidelity_acc) + '\n')
        all_un_screw_node_fidelity_prob = prob_un_screw_node
        print("all_un_screw_node_fidelity_prob:", all_un_screw_node_fidelity_prob)
        file_txt.write("all_un_screw_node_fidelity_prob:" + str(all_un_screw_node_fidelity_prob) + '\n') 
        Sparsity = 1 - torch.sum( node_mask* screw_all_mask) / torch.sum(All_one_mask).item()
        print("Sparsity：", format(Sparsity, '.4f'))
        file_txt.write("Sparsity：" + str(format(Sparsity, '.4f')) + '\n')

        return  all_un_screw_node_fidelity_acc,all_un_screw_node_fidelity_prob,Sparsity

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Dome for ST-GCN Explainer')
        parser.add_argument('--skeleton', default='S031C001P096R002A114',help='Path to video')
        parser.add_argument('--plot_action', default=False,
                            help='save action as image',type=bool)       
        parser.add_argument('--output_dir', default='./data/result',help='Path to save results')
        parser.add_argument('--exp_type', default='work1',help='one of explainer,pgm_explainer,GNN_Explainer')
        parser.add_argument('--data_level', default='test_set',help='instance or test_set')
        parser.add_argument('--valid', default='xsub',help='One of xsub and xview, and csub csetup')
        parser.add_argument('--feat_mask_type', default='individual_feature',help='individual_feature or cam')
        parser.add_argument('--skeleton_file_path',default='./data/nturgbd_raw/ntuall/', help='none')  
        parser.add_argument('--topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        args = parser.parse_known_args(namespace=parent_parser)
        parser.set_defaults(config='./config/st_gcnwork1/ntu-{}/test.yaml'.format(parent_parser.valid))
        parser.set_defaults(print_log=False)

        return parser