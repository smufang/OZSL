import torch
import ipdb
import logging
import os
import numpy as np
import torch.optim as optim
import torch.nn as nn
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import torch.nn.functional as F
import scipy.io
from scipy import io
import pickle
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
#from common import general
from sklearn.metrics import pairwise_distances
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from math import factorial
from utilis_svae import emd
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.metrics import f1_score
import sys

def norm_data(visual_features):
    for i in range(visual_features.shape[0]):
        visual_features[i,:] = visual_features[i,:]/np.linalg.norm(visual_features[i,:]) 
    return visual_features

def tsne_plot(data, target, file_name, use_marker=False):
    num_classes = np.unique(target).shape[0]

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(data)

    df = pd.DataFrame()
    df["y"] = target
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    scatterplot = plt.figure(figsize=(16,10))
    args = {'x': "comp-1", 
            'y': "comp-2", 
            'hue': df.y.tolist(),
            'palette': sns.color_palette("hls", num_classes),
            'data': df}
    if use_marker:
        args['style'] = df.y.tolist()
    
    sns.scatterplot(**args)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig = scatterplot.get_figure()
    fig.savefig("{}.png".format(file_name)) 
    plt.close()
    
def classifier(train_data, test_data, arch, n_epochs, is_dataloader=False, batch_size=512, define_classifier=True, predefined_cl5=None):
    if define_classifier:
        class cl5(nn.Module):
            def __init__(self, arch):
                super(cl5, self).__init__()
                self.model = nn.Sequential(*arch['model'])

            def forward(self, x):
                out = self.model(x)
                return out

        model = cl5(arch).cuda()
    else:
        model = predefined_cl5.cuda()

    if is_dataloader:
        train_dataloader = train_data
        test_dataloader = test_data
    else:
        train_feat = train_data[0]
        train_label = train_data[1]
        train_dataset = TensorDataset(torch.from_numpy(train_feat), torch.from_numpy(train_label))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_feat = test_data[0]
        test_label = test_data[1]
        test_dataset = TensorDataset(torch.from_numpy(test_feat), torch.from_numpy(test_label))
        test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=True)

    optimizer = optim.Adam(model.parameters(), **arch['optim'])

    for epoch in range(n_epochs):
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            model.eval()
            correct = 0
            total_test_samples = 0
            for i, (x, y) in enumerate(test_dataloader):
                x, y = x.cuda(), y.cuda()
                pred = torch.argmax(model(x), dim=1)
                correct += torch.sum(pred == y).detach().cpu().numpy()
                total_test_samples += y.size(0)
            print("test_acc: ", correct/total_test_samples)

            correct = 0
            total_test_samples = 0
            for i, (x, y) in enumerate(train_dataloader):
                x, y = x.cuda(), y.cuda()
                pred = torch.argmax(model(x), dim=1)
                correct += torch.sum(pred == y).detach().cpu().numpy()
                total_test_samples += y.size(0)
            print("train_acc: ", correct/total_test_samples)

    with torch.no_grad():
        model.eval()
        correct_dict = {i:0 for i in range(arch['num_classes'])}
        total_test_samples_dict = {i:0 for i in range(arch['num_classes'])}
        for i, (x, y) in enumerate(test_dataloader):
            x = x.cuda()
            y = y.detach().cpu().numpy()
            pred = torch.argmax(model(x), dim=1).detach().cpu().numpy()
            for j in range(len(y)):
                if y[j] == pred[j]:
                    correct_dict[y[j]] += 1
                total_test_samples_dict[y[j]] += 1
        print("class-wise accuracy:")
        MCA = []
        for j in range(arch['num_classes']):
            print('Class {}: {}'.format(j, correct_dict[j]/total_test_samples_dict[j]))
            MCA.append(correct_dict[j]/total_test_samples_dict[j])
        print('Mean Class Accuracy', np.mean(MCA))

def top1_accuracy(pred, target, total_target):
    unique_labels = np.unique(total_target)
    
    # Initialize the dict
    cls_wise_correct_smp = dict()
    cls_wise_total_smp = dict()
    for i in unique_labels:
        cls_wise_correct_smp[i] = 0
        cls_wise_total_smp[i] = 0
    
    # Compute Accuracy
    for i in range(len(pred)):
        if pred[i] == target[i]:
            cls_wise_correct_smp[target[i]] += 1
    
    for i in range(len(total_target)):
        cls_wise_total_smp[total_target[i]] += 1

    cls_wise_acc = []
    for i in unique_labels:
        cls_wise_acc.append(cls_wise_correct_smp[i]/cls_wise_total_smp[i])
    cls_wise_acc = np.array(cls_wise_acc)
    acc = np.mean(cls_wise_acc)

    return acc




class Model_train(object):
    def __init__(self, 
                 dataset_name,
                 encoder,
                 decoder,
                 attr_encoder,
                 attr_decoder,
                 classifier,
                 train_loader,
                 test_loader_unseen,
                 test_loader_seen,
                 criterion,
                 lr = 1e-3,
                 all_attrs = None,
                 epoch = 10000,
                 save_path = "/data/xingyu/wae_lle/experiments/",
                 save_every = 1,
                 iftest = False,
                 ifsample = False,
                 data = None,
                 GZSL = True,
                 zsl_classifier = None
                 ):  
        self.dataset_name = dataset_name
        self.encoder = encoder
        self.decoder = decoder
        self.attr_encoder = attr_encoder
        self.attr_decoder = attr_decoder
        self.classifier = classifier
        self.zsl_classifier = zsl_classifier
        self.train_loader = train_loader
        self.test_loader_unseen = test_loader_unseen
        self.test_loader_seen = test_loader_seen
           
        self.criterion = criterion
        self.crossEntropy_Loss = nn.NLLLoss()
        
        self.all_attrs = all_attrs
        self.lr = lr
        self.epoch = epoch
        self.save_path = save_path
        self.save_every = save_every
        self.ifsample = ifsample
        self.data = data
        self.GZSL = GZSL
        self.distribution = 'vmf'
        self.sinkhorn = emd.SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        
        if iftest:
            log_dir = '{}/log'.format(self.save_path)
            #general.logger_setup(log_dir, 'results__')
        
        
    def save_checkpoint(self,state, filename = 'checkpoint.pth.tar'):
        torch.save(state, filename)  
         
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
        else:
            raise NotImplemented

        return q_z
        
        
    
    def compute_acc(self,trues, preds):
        """
        Given true and predicted labels, computes average class-based accuracy.
        """

        # class labels in ground-truth samples
        classes = np.unique(trues)
        # class-based accuracies
        cb_accs = np.zeros(classes.shape, np.float32)
        #ipdb.set_trace()
        for i, label in enumerate(classes):
            inds_ci = np.where(trues == label)[0]

            cb_accs[i] = np.mean(
              np.equal(
              trues[inds_ci],
              preds[inds_ci]
            ).astype(np.float32)
        )
        #ipdb.set_trace()
        return np.mean(cb_accs)   
      
    def training(self, checkpoint = -1, checkpoint_num=None, save_path1=None):
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)
    
        if checkpoint_num != None:
            self.save_path1 = save_path1
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint_num)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint_num)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint_num)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint_num)
            #file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(checkpoint_num)
                
            enc_path = os.path.join(self.save_path1, file_encoder)
            dec_path = os.path.join(self.save_path1, file_decoder)
            attr_enc_path = os.path.join(self.save_path1, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path1, file_attr_decoder)
            #classifier_path = os.path.join(self.save_path, file_classifier)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
            
            #classifier_checkpoint = torch.load(classifier_path)
            #self.classifier.load_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.train()
        self.decoder.train()
        self.attr_encoder.train() 
        self.attr_decoder.train()
        self.classifier.train()
        
        enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
        classifier_optim = optim.Adam(self.classifier.parameters(), lr = self.lr)
              
        enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        classifier_scheduler = StepLR(classifier_optim, step_size=10000, gamma=0.5)
        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
            self.classifier = self.classifier.cuda()
        print("Begin Training ##############################>>>>>>>>")    
        for epoch in range(checkpoint+1, self.epoch):
            print("epoch: {}".format(epoch))
            # print("train_loader {}".format(len(self.train_loader)))
            # sys.exit()
            step = 0 
            train_data_iter = iter(self.train_loader)
            for i_batch, sample_batched in enumerate(self.train_loader):                      
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
                # print('unique_label', np.unique(input_label))
                # sys.exit()
              
                batch_size = input_data.size()[0]
                if torch.cuda.is_available():
                    input_data = input_data.float().cuda()
                    input_label = input_label.long().view(-1).cuda()
                    input_attr = input_attr.float().cuda().squeeze()
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                self.attr_decoder.zero_grad()
                self.classifier.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                
                z_x = z1.rsample()
                z_attr = z2.rsample()
                
                sub_batch_size = 10
                z_x_2 = z1.rsample(sub_batch_size).permute(1,0,2)
                z_attr_2 = z2.rsample(sub_batch_size).permute(1,0,2)
                
            
                z_input = torch.cat((z_attr.squeeze(), z_x),0) 
                label_input = torch.cat((input_label, input_label),0)
             
                cls_out = self.classifier(z_input)
                cls_loss = self.crossEntropy_Loss(cls_out, label_input) 
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                #original code
                x_recon = self.decoder(z_input)
                recon_loss = self.criterion(x_recon, torch.cat((input_data,input_data),0))
                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, torch.cat((input_attr,input_attr),0))
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
     
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)
                #ipdb.set_trace()
            
                KL_loss = dist.mean()
               
                total_loss =  recon_loss *1.0 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0  
            
                total_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()
                step += 1
            
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), cls_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))
   
            if epoch % self.save_every ==0: 
            
                file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict(), 
                     'optimizer': enc_optim.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict(), 
                     'optimizer': dec_optim.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': self.classifier.state_dict(), 
                     'optimizer': classifier_optim.state_dict()}, 
                     file_name_classifier)   
            
    
    def training_iter(self, checkpoint = -1, checkpoint_num=None, save_path1=None):
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)
    
        if checkpoint_num != None:
            self.save_path1 = save_path1
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint_num)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint_num)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint_num)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint_num)
            #file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(checkpoint_num)
                
            enc_path = os.path.join(self.save_path1, file_encoder)
            dec_path = os.path.join(self.save_path1, file_decoder)
            attr_enc_path = os.path.join(self.save_path1, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path1, file_attr_decoder)
            #classifier_path = os.path.join(self.save_path, file_classifier)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
            
            #classifier_checkpoint = torch.load(classifier_path)
            #self.classifier.load_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.train()
        self.decoder.train()
        self.attr_encoder.train() 
        self.attr_decoder.train()
        self.classifier.train()
        
        enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
        classifier_optim = optim.Adam(self.classifier.parameters(), lr = self.lr)
              
        enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        classifier_scheduler = StepLR(classifier_optim, step_size=10000, gamma=0.5)
        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
            self.classifier = self.classifier.cuda()
        print("Begin Training ##############################>>>>>>>>")    
        for epoch in range(checkpoint+1, self.epoch):
            print("epoch: {}".format(epoch))
            # print("train_loader {}".format(len(self.train_loader)))
            # sys.exit()
            step = 0 
            train_data_iter = iter(self.train_loader)
            for i_batch, sample_batched in enumerate(self.train_loader):                      
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
                # print('unique_label', np.unique(input_label))
                # sys.exit()
              
                batch_size = input_data.size()[0]
                if torch.cuda.is_available():
                    input_data = input_data.float().cuda()
                    input_label = input_label.long().view(-1).cuda()
                    input_attr = input_attr.float().cuda().squeeze()
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                self.attr_decoder.zero_grad()
                self.classifier.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                
                z_x = z1.rsample()
                z_attr = z2.rsample()
                
                sub_batch_size = 10
                z_x_2 = z1.rsample(sub_batch_size).permute(1,0,2)
                z_attr_2 = z2.rsample(sub_batch_size).permute(1,0,2)
                
            
                z_input = torch.cat((z_attr.squeeze(), z_x),0) 
                label_input = torch.cat((input_label, input_label),0)
             
                cls_out = self.classifier(z_input)
                cls_loss = self.crossEntropy_Loss(cls_out, label_input) 
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                #original code
                x_recon = self.decoder(z_input)
                recon_loss = self.criterion(x_recon, torch.cat((input_data,input_data),0))
                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, torch.cat((input_attr,input_attr),0))
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
     
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)
                #ipdb.set_trace()
            
                KL_loss = dist.mean()
               
                total_loss =  recon_loss *1.0 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0  
            
                total_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()
                step += 1
            
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), cls_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))

                if (step + 1) > 500:
                    break

            if epoch % self.save_every ==0: 
            
                file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict(), 
                     'optimizer': enc_optim.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict(), 
                     'optimizer': dec_optim.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': self.classifier.state_dict(), 
                     'optimizer': classifier_optim.state_dict()}, 
                     file_name_classifier)   
            
    
    def training_domain(self, cl2, checkpoint = -1, checkpoint_num=None, save_path1=None):
        print('Training Domain Loss:')
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)

        if checkpoint_num != None:
            self.save_path1 = save_path1
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint_num)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint_num)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint_num)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint_num)
            file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(checkpoint_num)
                
            enc_path = os.path.join(self.save_path1, file_encoder)
            dec_path = os.path.join(self.save_path1, file_decoder)
            attr_enc_path = os.path.join(self.save_path1, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path1, file_attr_decoder)
            classifier_path = os.path.join(self.save_path1, file_classifier)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
            
            classifier_checkpoint = torch.load(classifier_path)
            self.classifier.load_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.eval()
        for parm in self.encoder.parameters():
            parm.requires_grad = False
        self.decoder.eval()
        for parm in self.decoder.parameters():
            parm.requires_grad = False
        self.attr_encoder.train() 
        self.attr_decoder.train()
        self.classifier.train() # Train or Not Train?
        cl2.train()
        
        # enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        # dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
        classifier_optim = optim.Adam(self.classifier.parameters(), lr = self.lr)
        classifier_domain_optim = optim.Adam(cl2.parameters(), lr = self.lr)
              
        # enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        # dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        # attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        # attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        # classifier_scheduler = StepLR(classifier_optim, step_size=10000, gamma=0.5)
        
        unseen_data = self.data.Unseen_Data
        unseen_data_label = self.data.Unseen_Labels
        unseen_dataset = TensorDataset(torch.from_numpy(unseen_data).float(), torch.from_numpy(unseen_data_label).long())
        unseen_dataloader = DataLoader(unseen_dataset, shuffle=True, batch_size=128)
        full_attrs = torch.from_numpy(self.data.domain_attrs).float().cuda()
        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
            self.classifier = self.classifier.cuda()
            cl2 = cl2.cuda()
        print("Begin Training ##############################>>>>>>>>")    
        for epoch in range(checkpoint+1, self.epoch):
            print("epoch: {}".format(epoch))
            # print("train_loader {}".format(len(self.train_loader)))
            # sys.exit()
            step = 0 
            for i_batch, sample_batched in enumerate(self.train_loader):  
                unseen_data_iter = iter(unseen_dataloader)                    
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
                # print('unique_label', np.unique(input_label))
                # sys.exit()
              
                batch_size = input_data.size()[0]
                if torch.cuda.is_available():
                    input_data = input_data.float().cuda()
                    input_label = input_label.long().view(-1).cuda()
                    input_label_size = input_label.size(0)
                    input_attr = input_attr.float().cuda().squeeze()
                    input_data_unseen, input_label_unseen = next(unseen_data_iter)
                    input_data_unseen, input_label_unseen = input_data_unseen.cuda(), input_label_unseen.cuda()
                    input_attr_unseen = full_attrs[input_label_unseen]
                    
                    #input_data_classifier = torch.clone(input_data)
                    input_label_classifier = torch.clone(input_label)
                    
                    input_data = torch.cat((input_data, input_data_unseen), dim=0)
                    input_label = torch.cat((torch.zeros(input_label.size(0)), torch.ones(input_label_unseen.size(0))), dim=0).long().cuda()
                    input_attr = torch.cat((input_attr, input_attr_unseen), dim=0)
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                self.attr_decoder.zero_grad()
                self.classifier.zero_grad()
                cl2.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                
                z_x = z1.rsample()
                z_attr = z2.rsample()
                
                sub_batch_size = 10
                z_x_2 = z1.rsample(sub_batch_size).permute(1,0,2)
                z_attr_2 = z2.rsample(sub_batch_size).permute(1,0,2)
                
            
                # Normal Classification Loss
                z_input_classifier = torch.cat((z_attr[:input_label_size,:].squeeze(), z_x[:input_label_size,:]),0) 
                label_input_classifier = torch.cat((input_label_classifier, input_label_classifier),0)
             
                cls_out_classifier = self.classifier(z_input_classifier)
                cls_loss_classifier = self.crossEntropy_Loss(cls_out_classifier, label_input_classifier)
                
                
                # Domain Classification Loss
                z_input = torch.cat((z_attr.squeeze(), z_x),0) 
                label_input = torch.cat((input_label, input_label),0)
             
                cls_out = cl2(z_input)
                cls_loss = self.crossEntropy_Loss(cls_out, label_input) 
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                #original code
                x_recon = self.decoder(z_input)
                recon_loss = self.criterion(x_recon, torch.cat((input_data,input_data),0))
                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, torch.cat((input_attr,input_attr),0))
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
     
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)
                #ipdb.set_trace()
            
                KL_loss = dist.mean()
               
                total_loss =  recon_loss *1.0 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0 + cls_loss_classifier * 1.0 
            
                total_loss.backward()
            
                # enc_optim.step()
                # dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()
                classifier_domain_optim.step()
                step += 1
            
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), cls_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))
   
            if epoch % self.save_every ==0: 
            
                file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': cl2.state_dict(), 
                     'optimizer': classifier_optim.state_dict()}, 
                     file_name_classifier)   
    
    def training_domain_no_classifier(self, cl2, checkpoint = -1, checkpoint_num=None, save_path1=None):
        print('Training Domain Loss No Classifier:')
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)

        if checkpoint_num != None:
            self.save_path1 = save_path1
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint_num)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint_num)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint_num)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint_num)
            #file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(checkpoint_num)
                
            enc_path = os.path.join(self.save_path1, file_encoder)
            dec_path = os.path.join(self.save_path1, file_decoder)
            attr_enc_path = os.path.join(self.save_path1, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path1, file_attr_decoder)
            #classifier_path = os.path.join(self.save_path, file_classifier)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
            
            #classifier_checkpoint = torch.load(classifier_path)
            #self.classifier.load_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.eval()
        for parm in self.encoder.parameters():
            parm.requires_grad = False
        self.decoder.eval()
        for parm in self.decoder.parameters():
            parm.requires_grad = False
        self.attr_encoder.train() 
        self.attr_decoder.train()
        #self.classifier.train() # Train or Not Train?
        cl2.train()
        
        # enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        # dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
        # classifier_optim = optim.Adam(self.classifier.parameters(), lr = self.lr)
        classifier_domain_optim = optim.Adam(cl2.parameters(), lr = self.lr)
              
        # enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        # dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        # attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        # attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        # classifier_scheduler = StepLR(classifier_optim, step_size=10000, gamma=0.5)
        
        unseen_data = self.data.Unseen_Data
        unseen_data_label = self.data.Unseen_Labels
        unseen_dataset = TensorDataset(torch.from_numpy(unseen_data).float(), torch.from_numpy(unseen_data_label).long())
        unseen_dataloader = DataLoader(unseen_dataset, shuffle=True, batch_size=128)
        full_attrs = torch.from_numpy(self.data.domain_attrs).float().cuda()
        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
            #self.classifier = self.classifier.cuda()
            cl2 = cl2.cuda()
        print("Begin Training ##############################>>>>>>>>")    
        for epoch in range(checkpoint+1, self.epoch):
            print("epoch: {}".format(epoch))
            # print("train_loader {}".format(len(self.train_loader)))
            # sys.exit()
            step = 0 
            for i_batch, sample_batched in enumerate(self.train_loader):  
                unseen_data_iter = iter(unseen_dataloader)                    
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
                # print('unique_label', np.unique(input_label))
                # sys.exit()
              
                batch_size = input_data.size()[0]
                if torch.cuda.is_available():
                    input_data = input_data.float().cuda()
                    input_label = input_label.long().view(-1).cuda()
                    # input_label_size = input_label.size(0)
                    input_attr = input_attr.float().cuda().squeeze()
                    input_data_unseen, input_label_unseen = next(unseen_data_iter)
                    input_data_unseen, input_label_unseen = input_data_unseen.cuda(), input_label_unseen.cuda()
                    input_attr_unseen = full_attrs[input_label_unseen]
                    
                    # input_label_classifier = torch.clone(input_label)
                    
                    input_data = torch.cat((input_data, input_data_unseen), dim=0)
                    input_label = torch.cat((torch.zeros(input_label.size(0)), torch.ones(input_label_unseen.size(0))), dim=0).long().cuda()
                    input_attr = torch.cat((input_attr, input_attr_unseen), dim=0)
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                self.attr_decoder.zero_grad()
                #self.classifier.zero_grad()
                cl2.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                
                z_x = z1.rsample()
                z_attr = z2.rsample()
                
                sub_batch_size = 10
                z_x_2 = z1.rsample(sub_batch_size).permute(1,0,2)
                z_attr_2 = z2.rsample(sub_batch_size).permute(1,0,2)
                
            
                # Normal Classification Loss
                # z_input_classifier = torch.cat((z_attr.squeeze()[input_label_size], z_x[input_label_size]),0) 
                # label_input_classifier = torch.cat((input_label_classifier, input_label_classifier),0)
             
                # cls_out_classifier = self.classifier(z_input_classifier)
                # cls_loss_classifier = self.crossEntropy_Loss(cls_out_classifier, label_input_classifier)
                
                
                # Domain Classification Loss
                z_input = torch.cat((z_attr.squeeze(), z_x),0) 
                label_input = torch.cat((input_label, input_label),0)
             
                cls_out = cl2(z_input)
                cls_loss = self.crossEntropy_Loss(cls_out, label_input) 
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                #original code
                x_recon = self.decoder(z_input)
                recon_loss = self.criterion(x_recon, torch.cat((input_data,input_data),0))
                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, torch.cat((input_attr,input_attr),0))
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
     
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)
                #ipdb.set_trace()
            
                KL_loss = dist.mean()
               
                total_loss =  recon_loss *1.0 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0  # + cls_loss_classifier * 1.0 
            
                total_loss.backward()
            
                # enc_optim.step()
                # dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                #classifier_optim.step()
                classifier_domain_optim.step()
                step += 1
            
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), cls_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))
   
            if epoch % self.save_every ==0: 
            
                file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': cl2.state_dict(), 
                     'optimizer': classifier_domain_optim.state_dict()}, 
                     file_name_classifier)   
    
    
    def training_cosine(self, checkpoint = -1, checkpoint_num=None, save_path1=None):
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)
    
        if checkpoint_num != None:
            self.save_path1 = save_path1
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint_num)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint_num)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint_num)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint_num)
            #file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(checkpoint_num)
                
            enc_path = os.path.join(self.save_path1, file_encoder)
            dec_path = os.path.join(self.save_path1, file_decoder)
            attr_enc_path = os.path.join(self.save_path1, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path1, file_attr_decoder)
            #classifier_path = os.path.join(self.save_path, file_classifier)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
            
            #classifier_checkpoint = torch.load(classifier_path)
            #self.classifier.load_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.eval()
        for parm in self.encoder.parameters():
            parm.requires_grad = False
        self.decoder.eval()
        for parm in self.decoder.parameters():
            parm.requires_grad = False
        self.attr_encoder.train() 
        self.attr_decoder.train()
        self.classifier.train()
        
        # enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        # dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
        classifier_optim = optim.Adam(self.classifier.parameters(), lr = self.lr)
              
        # enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        # dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        classifier_scheduler = StepLR(classifier_optim, step_size=10000, gamma=0.5)
        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
            self.classifier = self.classifier.cuda()
            attr_anchor = torch.from_numpy(self.data.attrs).float().cuda()
        print("Begin Training Cosine ##############################>>>>>>>>")    
        for epoch in range(checkpoint+1, self.epoch):
            print("epoch: {}".format(epoch))
            # print("train_loader {}".format(len(self.train_loader)))
            # sys.exit()
            step = 0 
            # train_data_iter = iter(self.train_loader)
            for i_batch, sample_batched in enumerate(self.train_loader):                      
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
                # print('unique_label', np.unique(input_label))
                # sys.exit()
              
                batch_size = input_data.size()[0]
                if torch.cuda.is_available():
                    input_data = input_data.float().cuda()
                    input_label = input_label.long().view(-1).cuda()
                    input_attr = input_attr.float().cuda().squeeze()
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                self.attr_decoder.zero_grad()
                self.classifier.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                
                z_x = z1.rsample()
                z_attr = z2.rsample()
                
                sub_batch_size = 10
                z_x_2 = z1.rsample(sub_batch_size).permute(1,0,2)
                z_attr_2 = z2.rsample(sub_batch_size).permute(1,0,2)
                
            
                z_input = torch.cat((z_attr.squeeze(), z_x),0) 
                label_input = torch.cat((input_label, input_label),0)
             
                # cls_out = self.classifier(z_input)
                # cls_loss = self.crossEntropy_Loss(cls_out, label_input) 

                # Cosine Similarity
                m_anchor, s_anchor = self.attr_encoder(attr_anchor)
                z_anchor = self.reparameterize(m_anchor, s_anchor)
                z_anchor = z_anchor.rsample()
                
                z_anchor = F.normalize(z_anchor, dim=1)
                z_feat = F.normalize(z_x, dim=1)

                cosine_similarity = z_feat @ z_anchor.T

                # print(cosine_similarity.shape)
                
                # Contrastive Loss
                # cls_loss = F.cross_entropy(cosine_similarity, input_label)
                
                # Triplet Loss
                p_mask = torch.zeros(cosine_similarity.size()).cuda()
                p_mask[:, input_label] = 1

                n_mask = torch.ones(cosine_similarity.size()).cuda()
                n_mask[:, input_label] = 0

                cls_loss = F.relu(0.8*(40-1) - (cosine_similarity*p_mask*(40-1)).sum(dim=1) + (cosine_similarity*n_mask).sum(dim=1))
                cls_loss = cls_loss.sum() / (len(input_label) * 40)
                
                # Triplet Loss 2:
                # triplet_loss = nn.TripletMarginLoss()
                # cls_loss = triplet_loss()
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                #original code
                x_recon = self.decoder(z_input)
                recon_loss = self.criterion(x_recon, torch.cat((input_data,input_data),0))
                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, torch.cat((input_attr,input_attr),0))
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
     
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)
                #ipdb.set_trace()
            
                KL_loss = dist.mean()
               
                total_loss =  recon_loss *1.0 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0  
            
                total_loss.backward()
            
                # enc_optim.step()
                # dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()
                step += 1
            
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), cls_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))
   
            if epoch % self.save_every ==0: 
            
                file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': self.classifier.state_dict(), 
                     'optimizer': classifier_optim.state_dict()}, 
                     file_name_classifier)   
    
    
    
    
    def training2(self, checkpoint = -1, checkpoint_num=None, save_path1=None):
        print('*******************************************************************')
        print('Training2')
        print('*******************************************************************')
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)
    
        if checkpoint_num != None:
            print('*******************************************************************')
            print('Previous Weight')
            print('*******************************************************************')
            self.save_path1 = save_path1
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint_num)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint_num)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint_num)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint_num)
            #file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(checkpoint_num)
                
            enc_path = os.path.join(self.save_path1, file_encoder)
            dec_path = os.path.join(self.save_path1, file_decoder)
            attr_enc_path = os.path.join(self.save_path1, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path1, file_attr_decoder)
            #classifier_path = os.path.join(self.save_path, file_classifier)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
            
            #classifier_checkpoint = torch.load(classifier_path)
            #self.classifier.load_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.train()
        self.decoder.train()
        self.attr_encoder.train() 
        self.attr_decoder.train()
        self.classifier.train()
        
        enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
        classifier_optim = optim.Adam(self.classifier.parameters(), lr = self.lr)
              
        enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        classifier_scheduler = StepLR(classifier_optim, step_size=10000, gamma=0.5)
        
        
        # Take seen data as OOD:
        seen_ood_feat = np.vstack([self.data.train_set_, self.data.val_set])
        seen_ood_labels = np.asarray([5 for i in range(len(seen_ood_feat))]) # np.vstack([self.data.train_labels_, self.data.val_labels])
        
        seen_ood_dataset = TensorDataset(torch.from_numpy(seen_ood_feat), torch.from_numpy(seen_ood_labels))
        seen_ood_dataloader = DataLoader(seen_ood_dataset, shuffle=True, batch_size=128)
        

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
            self.classifier = self.classifier.cuda()
        print("Begin Training ##############################>>>>>>>>")    
        for epoch in range(checkpoint+1, self.epoch):
            print("epoch: {}".format(epoch))
            # print("train_loader {}".format(len(self.train_loader)))
            # sys.exit()
            step = 0 
            train_data_iter = iter(self.train_loader)
            for i_batch, sample_batched in enumerate(self.train_loader):                      
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
                # print('unique_label', np.unique(input_label))
                # sys.exit()

                ood_iter = iter(seen_ood_dataloader)
                ood_feat, ood_label = next(ood_iter)
              
                batch_size = input_data.size()[0]
                if torch.cuda.is_available():
                    input_data = input_data.float().cuda()
                    input_label = input_label.long().view(-1).cuda()
                    input_attr = input_attr.float().cuda().squeeze()
                    ood_feat, ood_label = ood_feat.float().cuda(), ood_label.long().cuda()
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                self.attr_decoder.zero_grad()
                self.classifier.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                
                # Seen as OOD
                m1_ood, s1_ood = self.encoder(ood_feat)
                z1_ood = self.reparameterize(m1_ood, s1_ood)
                z_ood = z1_ood.rsample()

                z_x = z1.rsample()
                z_attr = z2.rsample()
                
                sub_batch_size = 10
                z_x_2 = z1.rsample(sub_batch_size).permute(1,0,2)
                z_attr_2 = z2.rsample(sub_batch_size).permute(1,0,2)
                
            
                z_input = torch.cat((z_attr.squeeze(), z_x),0) 
                label_input = torch.cat((input_label, input_label),0)
             
                cls_out = self.classifier(z_input)
                cls_out_ood = self.classifier(z_ood)
                cls_loss = self.crossEntropy_Loss(cls_out, label_input) + 0.5 * self.crossEntropy_Loss(cls_out_ood, ood_label)
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                #original code
                x_recon = self.decoder(z_input)
                x_recon_ood = self.decoder(z_ood)
                recon_loss = self.criterion(x_recon, torch.cat((input_data,input_data),0)) + self.criterion(x_recon_ood, ood_feat)
                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, torch.cat((input_attr,input_attr),0))
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
     
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)
                #ipdb.set_trace()
            
                KL_loss = dist.mean()
               
                total_loss =  recon_loss *1.0 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0  
            
                total_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()
                step += 1
            
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), cls_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))
   
            if epoch % self.save_every ==0: 
            
                file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict(), 
                     'optimizer': enc_optim.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict(), 
                     'optimizer': dec_optim.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': self.classifier.state_dict(), 
                     'optimizer': classifier_optim.state_dict()}, 
                     file_name_classifier) 
    
    
    
    def search_thres_by_sample(self, attrs, n = 10000):
        min_thres = 100
        m, s = self.attr_encoder(attrs)
      
        z = []
        for i in range(n):
            z_fake = self.reparameterize(m, s).rsample()
            dist = F.cosine_similarity(m, z_fake)
            z.append(z_fake)
            thres = dist.min()
            if min_thres > thres:
                min_thres = thres
        
        return min_thres
        
    def load_models(self, epoch, setting=None):
        file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
        file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
        file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)  
        file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)  
        enc_path = os.path.join(self.save_path, file_encoder)
        dec_path = os.path.join(self.save_path, file_decoder)
        attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
        classifier_path = os.path.join(self.save_path, file_classifier)
        enc_checkpoint = torch.load(enc_path)
        self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        dec_checkpoint = torch.load(dec_path)
        self.decoder.load_state_dict(dec_checkpoint['state_dict'])
        attr_enc_checkpoint = torch.load(attr_enc_path)
        self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
        
        if setting == 2:
            # save_path = '{}_{}_{}'.format('/home/sethupathy/openworl_zsl/gzsl_svae/experiments/AWA2',512, 64)
            save_path = '{}_{}_{}'.format('/home/sethupathy/openworl_zsl/gzsl_svae/experiments/AWA2_synthetic_cvae300',512, 64)
            classifier_path = os.path.join(save_path, file_classifier)
            classifier_checkpoint = torch.load(classifier_path)
            print("i am here")
        else:
            classifier_checkpoint = torch.load(classifier_path)
        self.classifier.load_state_dict(classifier_checkpoint['state_dict'])       
        
        # Load the ZSL classifiers. These ZSL classifiers can be replaced by any SOTA models! 
        if self.dataset_name == 'AWA1':
            zsl_classifier_checkpoint = torch.load("/home/sethupathy/openworl_zsl/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/zsl_models/awa1_Checkpoint_24_Classifier.pth.tar")
        elif self.dataset_name == 'AWA2':
            zsl_classifier_checkpoint = torch.load("/home/sethupathy/openworl_zsl/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/zsl_models/awa2_Checkpoint_9_Classifier.pth.tar")
        elif self.dataset_name == 'CUB':
            zsl_classifier_checkpoint = torch.load("/home/sethupathy/openworl_zsl/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/zsl_models/cub_Checkpoint_7_Classifier.pth.tar")
        elif self.dataset_name == 'FLO':
            zsl_classifier_checkpoint = torch.load("/home/sethupathy/openworl_zsl/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/zsl_models/flo_Checkpoint_24_Classifier.pth.tar")
        elif self.dataset_name == 'SUN':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/sun/Checkpoint_14_Classifier.pth.tar")
        
        self.zsl_classifier.load_state_dict(zsl_classifier_checkpoint['state_dict'])
        
        self.encoder.eval()
        self.decoder.eval()
        self.attr_encoder.eval()  
        self.zsl_classifier.eval()      
        if torch.cuda.is_available():
             self.encoder, self.decoder, self.attr_encoder, self.zsl_classifier, self.classifier = self.encoder.cuda(), self.decoder.cuda(), self.attr_encoder.cuda(), self.zsl_classifier.cuda(), self.classifier.cuda()
        
    
     
    def search_thres_by_traindata(self, epoch, dataset = None, n = 0.95):
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        # seen_labels = dataset.seen_labels
        # unseen_labels = dataset.unseen_labels
        self.load_models(epoch)

        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []    
        all_anchors = self.attr_encoder(all_attrs)[0]      
        # seen_idx = seen_labels - 1
        # unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors #all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]       
        seen_count = 0
        seen_all = 0
        unseen_count = 0
        unseen_all = 0
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        
        for i_batch, sample_batched in enumerate(self.train_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            #z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                kk = input_label[k]+1 #input_label[k,:]+1
                z_tile = z_real[k,:].repeat(seen_anchors.shape[0]).view(seen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, seen_anchors)
                if min_thres>dist.max():
                    min_thres = dist.max()
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
            
        dist_array = np.array(dist_list)
        idx = dist_array.shape[0] * (1.0 - n)
        thres  = np.sort(dist_array)[int(idx)]

      
        return thres 
    
    def testing_1(self, epoch, test_class = 'seen', dataset = None, D = None, threshold = 0.99):
        coeff_fun_map = {
            'optimized_seq':
            lambda data, dic: cpp.decomp_simplex_sequence(
                data, dic, n_smooth_iter=1, sub_window_size=3, lambda1=0.1),
            'optimized': cpp.decomp_simplex
        }
        coeff_fun = coeff_fun_map['optimized']
        if test_class == 'seen':
            test_loader = self.test_loader_seen
        elif test_class == 'unseen':
            test_loader = self.test_loader_unseen
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = dataset.seen_labels
        unseen_labels = dataset.unseen_labels
        
        if isinstance(threshold, np.ndarray):
            thresholds = threshold
        else:
            thresholds = np.ones(seen_labels.shape[0]) * threshold
            
        self.load_models(epoch) 
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels - 1
        unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        
        seen_count = 0
        seen_all = 1
        unseen_count = 0
        unseen_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []
        for i_batch, sample_batched in enumerate(test_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]           
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]+1
                gt.append(kk.data.item()-1)
                #ipdb.set_trace()
                dist = []
                for jj in range(len(D)):
                    z_k = z_real[k,:].cpu().data.numpy().astype('float64').T.reshape(-1,1)
                    coeff_k = coeff_fun(z_k, D[jj])
                    z_recon = np.dot(D[jj], coeff_k)
                    error = np.linalg.norm(z_k - z_recon)
                    dist.append(error)
                min_dist = np.array(dist).min()
                dist_list.append(min_dist)
                all_count += 1
                #print('processing ')  
               
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if min_dist>threshold: 
                        out = self.zsl_classifier(input_k.view(1,-1))
                        pred_label_ = torch.argmax(out,1)
                        pred_label = self.data.unseen_labels[pred_label_.cpu().data.item()]-1
                        pred.append(pred_label)
                        unseen_count +=1
                    else:
                        pred.append(1000)
                    
                    
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1            
                    if min_dist<=threshold:
                        seen_count +=1
                        out = self.classifier(z_real[k,:].view(1,-1))
                        pred_label = torch.argmax(out,1).data.item()
                        #pred_label = self.data.test_seen_labels[pred_label_.cpu().data.item()]-1
                        pred.append(pred_label)
                    else:
                        pred.append(1000) 
        pred_ = np.vstack(pred)
        gt_ = np.vstack(gt)
        acc = self.compute_acc(gt_, pred_ )

        mean_dist = mean_dist /all_count
        #ipdb.set_trace()
        return unseen_count/unseen_all, seen_count/seen_all , acc, dist_list
               
    def testing_2(self, epoch, test_class = 'seen', dataset = None, threshold = 0.99):
        
        if test_class == 'seen':
            test_loader = self.test_loader_seen
        elif test_class == 'unseen':
            test_loader = self.test_loader_unseen
        
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = dataset.seen_labels
        unseen_labels = dataset.unseen_labels
        
        if isinstance(threshold, np.ndarray):
            thresholds = threshold
        else:
            thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels - 1
        unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        
        seen_count = 0
        seen_all = 1
        unseen_count = 0
        unseen_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []
        for i_batch, sample_batched in enumerate(test_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]           
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]+1
                gt.append(kk.data.item()-1)
                z_tile = z_real[k,:].repeat(seen_anchors.shape[0]).view(seen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, seen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        out = self.zsl_classifier(input_k.view(1,-1))
                        pred_label_ = torch.argmax(out,1)
                        pred_label = self.data.unseen_labels[pred_label_.cpu().data.item()]-1
                        pred.append(pred_label)
                        unseen_count +=1
                    else:
                        pred.append(1000)
                    
                    
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1            
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1
                        out = self.classifier(z_real[k,:].view(1,-1))
                        pred_label = torch.argmax(out,1).data.item()
                        #pred_label = self.data.test_seen_labels[pred_label_.cpu().data.item()]-1
                        pred.append(pred_label)
                    else:
                        pred.append(1000)
               
                                        
        pred_ = np.vstack(pred)
        gt_ = np.vstack(gt)
        acc = self.compute_acc(gt_, pred_ )

        mean_dist = mean_dist /all_count
        return unseen_count/unseen_all, seen_count/seen_all , acc, dist_list
        
    def draw_roc_curve(self, epoch, data):
        import sklearn.metrics as metrics
        unseen_acc, _, ts, dist_unseen = self.testing_2(epoch, test_class ='unseen', dataset = data, threshold = 0.63)
        _, seen_acc,  tr, dist_seen = self.testing_2(epoch, test_class ='seen', dataset = data, threshold = 0.63)
         
        print('fpr = {}, tpr = {}'.format(1-unseen_acc, seen_acc)) 
        ipdb.set_trace()
        dists = np.concatenate((np.array(dist_unseen), np.array(dist_seen)))
        
        labels_unseen = np.zeros(len(dist_unseen))
        labels_seen = np.ones(len(dist_seen))
        
        labels = np.concatenate((labels_unseen, labels_seen))
        fpr, tpr, threshold = metrics.roc_curve(labels, dists)
        roc_auc = metrics.auc(fpr, tpr)
        
        #print('fpr = {}, tpr = {}, auc = {}'.format(1-fpr, tpr, roc_auc))
        
        with open("{}_res.pkl".format(self.dataset_name), 'wb') as f:      
            pickle.dump({'fpr': fpr, 'tpr':tpr}, f)  
            f.close()
            print('save data done!')
            
        plt.title('ROC curves on the 5 benchmark datasets')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        ipdb.set_trace()
        return 0 
        
    
    
    def testing(self, epoch, if_viz = True, sample_rate = 2):
        file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
        file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
        file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
        
        enc_path = os.path.join(self.save_path, file_encoder)
        dec_path = os.path.join(self.save_path, file_decoder)
        attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
    
        enc_checkpoint = torch.load(enc_path)
        self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
        dec_checkpoint = torch.load(dec_path)
        self.decoder.load_state_dict(dec_checkpoint['state_dict'])
        
        attr_enc_checkpoint = torch.load(attr_enc_path)
        self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
         
        self.encoder.eval()
        self.decoder.eval()
        self.attr_encoder.eval()
        
        if torch.cuda.is_available():
             self.encoder, self.decoder, self.attr_encoder = self.encoder.cuda(), self.decoder.cuda(), self.attr_encoder.cuda()
             
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        '''
        class_names = ["antelope", "grizzly bear", "killer whale", "beaver", "dalmatian", "persian cat", "horse",
                           "german shepherd", "blue whale", "siamese cat", "skunk",  "mole", "tiger", "hippopotamus",
                           "leopard", "moose", "spider monkey", "humpback whale", "elephant", "gorilla", "ox",  "fox",
                           "sheep", "seal" ,"chimpanzee", "hamster", "squirrel", "rhinoceros", "rabbit", "bat", "giraffe",
                           "wolf", "chihuahua", "rat", "weasel","otter", "buffalo", "zebra", "giant panda", "deer", "bobcat",
                           "pig", "lion", "mouse", "polar bear", "collie", "walrus", "raccoon", "cow", "dolphin"]
        '''
        class_names = ["Seen Features","Unseen Features"]                   
        
        for i_batch, sample_batched in enumerate(self.test_loader_unseen):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]
            
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()
            
            
            if self.ifsample:
                m, s = self.encoder(input_data)
                z_real = self.reparametrize(m, s)
            else:
                z_real = self.encoder(input_data)[0]
            
            x_recon = self.decoder(z_real)
                
            mu, sigma = self.attr_encoder(input_attr)
            z_fake = self.reparameterize(mu, sigma).rsample().squeeze()
       
            muu.append(z_fake.squeeze().cpu().data.numpy())
            
            z.append(z_real.cpu().data.numpy())
            label.append(input_label.cpu().data.numpy().reshape(-1,1))
            recon.append(x_recon.cpu().data.numpy())
            data_in.append(input_data.cpu().data.numpy())
            z_attr.append(z_fake.squeeze().cpu().data.numpy())
            
            recon_loss = self.criterion(x_recon, input_data)
            recon_loss = torch.dot(z_real[1,:], z_fake[1,:])
            print('batch {} recon_loss = {}'.format(i_batch, recon_loss))
        
   
        muu_ = np.vstack(muu)      
        z_ = np.vstack(z)
        recon_ = np.vstack(recon)
        label_ = np.vstack(label).reshape(-1)
        data_in_ = np.vstack(data_in)
        z_attr_ = np.vstack(z_attr)
      
        if if_viz:
            from sklearn.manifold import TSNE
            from matplotlib import colors as mcolors

            colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
            color_list = []
            for color in colors.keys():
                if color == 'aliceblue':
                    color_list.append('y')
                elif color == 'k':
                    color_list.append('purple')
                else:
                    color_list.append(color)
            
            color_list[0] = 'blue'
            color_list[1] = 'darkorange'
            
            label_colors = []
            label_names = []
     
            for i in range(len(z_attr_)):
                label_colors.append(color_list[label_[i]])
                label_names.append(class_names[label_[i]])
          
        
            model = TSNE(n_components = 2, n_iter = 5000, init = 'pca',random_state = 0)
               
            #zz_ = np.vstack([z_, muu_])
            zz_ = np.vstack([z_, z_])
            label_colors__ = label_colors
            label_colors = label_colors + label_colors__      
            z_sample = zz_[range(0,zz_.shape[0],sample_rate),:]
            label_colors_sample = label_colors[::sample_rate] 
            label_names_sample = label_names[::sample_rate] 

            z_2d = model.fit_transform(z_sample)
            fig = plt.figure(figsize = (12, 12) )
            ax = fig.add_subplot(111)
            n = z_2d.shape[0]
            
            
            df1 = pd.DataFrame({"x": z_2d[0:n//2, 0], "y": z_2d[0:n//2, 1], "colors": label_colors_sample[0:n//2]})
            for i, dff in df1.groupby("colors"):
                class_name = class_names[color_list.index(i)]
                plt.scatter(dff['x'], dff['y'], c=i, label= class_name, marker = '.')
              
            ax.scatter(z_2d[0:n//2, 0], z_2d[0:n//2, 1], c=label_colors_sample[0:n//2] , marker = '.')
            ax.scatter(z_2d[n//2:n, 0], z_2d[n//2:n, 1], c=label_colors_sample[n//2:n], marker = '.')
            ax.set_facecolor('gray')
            #ax.set_ylim(-48, 48)
            #ax.set_xlim(-48, 48)
            plt.axis('off')
            #ax.set_yticklabels([])
            #ax.set_xticklabels([])
            
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
            ax.legend(fontsize = 'xx-large',loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
            #plt.legend(fontsize = "small", loc=1)
            plt.show()
            ipdb.set_trace()
        return z_, recon_, label 

    def testing_split(self, epoch, test_class = 'seen', dataset = None, threshold = 0.99):
        
        if test_class == 'seen':
            test_loader = self.test_loader_seen
        elif test_class == 'unseen':
            test_loader = self.test_loader_unseen
        
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = dataset.seen_labels
        unseen_labels = dataset.unseen_labels
        
        if isinstance(threshold, np.ndarray):
            thresholds = threshold
        else:
            thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels - 1
        unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        
        seen_count = 0
        seen_all = 1
        unseen_count = 0
        unseen_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        seen_feat_split = []
        seen_label_gt = []
        unseen_feat_split = []
        unseen_label_gt = []
        
        for i_batch, sample_batched in enumerate(test_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]           
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]+1
                gt.append(kk.data.item()-1)
                z_tile = z_real[k,:].repeat(seen_anchors.shape[0]).view(seen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, seen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        if len(unseen_feat_split) == 0:
                            unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                            unseen_label_gt = input_label[k].detach().cpu().numpy() - 1

                        else:
                            unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                            unseen_label_gt = np.concatenate((unseen_label_gt, input_label[k].detach().cpu().numpy() - 1), axis=0)
                        
                        unseen_count +=1
                    else:
                        if len(seen_feat_split) == 0:
                            seen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                            seen_label_gt = input_label[k].detach().cpu().numpy() - 1

                        else:
                            seen_feat_split = np.concatenate((seen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                            seen_label_gt = np.concatenate((seen_label_gt, input_label[k].detach().cpu().numpy() - 1), axis=0)
                    
                    
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1            
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1
                        
                        if len(seen_feat_split) == 0:
                            seen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                            seen_label_gt = input_label[k].detach().cpu().numpy() - 1

                        else:
                            seen_feat_split = np.concatenate((seen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                            seen_label_gt = np.concatenate((seen_label_gt, input_label[k].detach().cpu().numpy() - 1), axis=0)
                    
                    else:
                        if len(unseen_feat_split) == 0:
                            unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                            unseen_label_gt = input_label[k].detach().cpu().numpy() - 1

                        else:
                            unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                            unseen_label_gt = np.concatenate((unseen_label_gt, input_label[k].detach().cpu().numpy() - 1), axis=0)
               
                                        
        print('unseen_perc: ', unseen_count/unseen_all, 'seen_prec: ', seen_count/seen_all)
        data_dict = {}
        data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(test_class))
            

    def testing_split_ood(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_unseen.pt')
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
        seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1
        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
            else:
                seen_label[i] = 45

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
            else:
                unseen_label[i] = 45

        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(unseen_feat), torch.from_numpy(unseen_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        for i_batch, (input_data, input_label) in enumerate(test_loader):
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                # input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]
                gt.append(kk.data.item())
                z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, unseen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if dist.max()<thresholds[max_idx]: 
                    if len(ood_feat_split) == 0:
                        ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        # print("ood_label_gt ", ood_label_gt)
                        # print("input_label ", input_label[k].detach().cpu().numpy())
                        ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
                    unseen_count +=1
                else:
                    if len(unseen_feat_split) == 0:
                        unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
                    
        total_ood = len(gt_label[gt_label == 45])
        ood_split = len(ood_label_gt[ood_label_gt == 45])
        print("ood_percentage: ", ood_split/total_ood)

        total_unseen = len(gt_label[np.logical_and(gt_label < 45, gt_label >= 40)])
        unseen_split = len(unseen_label_gt[np.logical_and(unseen_label_gt < 45, unseen_label_gt >= 40)])
        print("unseen_percentage: ", unseen_split/total_unseen)
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))
        
        
    def testing_split_ood_vis(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_unseen.pt')
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
        seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1
        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
            else:
                seen_label[i] = 45

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
            else:
                unseen_label[i] = 45

        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        all_test_data = np.concatenate((seen_feat, unseen_feat), axis=0)
        data_feat = torch.from_numpy(all_test_data[gt_label == 40]) #torch.from_numpy(unseen_feat)
        label_feat = torch.from_numpy(gt_label[gt_label == 40]) #torch.from_numpy(unseen_label)
        #all_test_data = torch.from_numpy((seen_feat, unseen_feat))

        # test_dataset = TensorDataset(data_unseen, label_unseen)
        # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        
        # for i_batch, (input_data, input_label) in enumerate(test_loader):
        #     if torch.cuda.is_available():
        #         input_data = input_data.float().cuda()
        #         input_label = input_label.cuda()  
                                                
        #     m, s = self.encoder(input_data)   
        #     z_real = self.reparameterize(m, s).rsample().squeeze()
        #     z_real = m.squeeze()
        
        ####################################################################
        # input_data = data_feat.float().cuda()
        # m, s = self.encoder(input_data)   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()
        # z_real = z_real.detach().cpu().numpy()
        # label_feat = label_feat.detach().cpu().numpy()
        # #label_feat = np.asarray([0 for i in range(len(seen_label))] + [1 for i in range(len(unseen_label))])
        # tsne_plot(z_real, label_feat, 'unseen_data')
        # #input_data.detach().cpu().numpy()
        ####################################################################

        ######################################################################
        #input_data = data_feat.float().cuda()
        bool_loc = np.logical_and(gt_label >= 40, gt_label < 45) #gt_label < 40 #np.logical_and(gt_label >= 40, gt_label < 45)
        bool_loc2 = gt_label == 45 #np.logical_and(gt_label >= 40, gt_label < 45) #gt_label == 45
        input_data2 = torch.from_numpy(all_test_data[bool_loc]).float().cuda()
        input_data3 = torch.from_numpy(all_test_data[bool_loc2]).float().cuda()
        len_per_class = len(all_test_data[gt_label == 40])

        #######################################
        # CVAE Synthetic Features
        checkpoint = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/cvae_attr_refinement/cub/syn_data.pt')
        syn_feat = checkpoint["syn_feat"]
        syn_label = np.asarray(checkpoint["syn_label"])
        
        print(np.unique(syn_label))
        print(known_unseen_classes)
        print(known_unseen_classes)
        print(syn_feat.shape)


        known_unseen_idx = [i for i in range(len(syn_feat)) if syn_label[i] in known_unseen_classes]
        known_syn_feat = torch.from_numpy(syn_feat[known_unseen_idx]).float().cuda()
        known_syn_label = torch.from_numpy(np.array([known_unseen_classes.index(syn_label[known_unseen_idx[i]]) for i in range(len(known_unseen_idx))]))
        print(np.unique(syn_label[known_unseen_idx]))
        #print(np.unique(known_syn_label))
        print('known_unseen_clasees', np.unique(known_unseen_classes))
        #######################################

        #######################################
        # Checkpoint gsm
        checkpoint_gsm = torch.load('/home/sethupathy/openworl_zsl/GSMFlow-main/awa2_syn_gsmflow.pt')
        gsm_syn_feat = checkpoint_gsm['gen_feat']
        gsm_syn_label = checkpoint_gsm['gen_label']
        #gsm_syn_feat_idx = np.random.choice(len(checkpoint_gsm['gen_feat']), 300*10, replace=False)
        #gsm_syn_feat = torch.from_numpy(gsm_syn_feat[gsm_syn_feat_idx]).float().cuda()
        gsm_syn_feat = torch.from_numpy(gsm_syn_feat).float().cuda()
        #gsm_syn_feat = torch.from_numpy(gsm_syn_feat).float().cuda()
        gsm_syn_label = torch.from_numpy(checkpoint_gsm['gen_label']).long().cuda()
        #######################################

        attrs = all_attrs[unseen_labels].repeat(200, 1)#.view(data_feat.size(0)*5,-1)
        svae_label = torch.from_numpy(np.asarray([unseen_labels.tolist().index(i) for i in unseen_labels])).long().cuda().repeat(200)
        m, s = self.attr_encoder(attrs)
        z_real = self.reparameterize(m, s).rsample().squeeze()
        z_real = self.decoder(z_real)
        svae_data = {'feat': z_real, 'label': svae_label}
        torch.save(svae_data, './synthetic_unseen_svae.zip')
        sys.exit()
        #z_real = z_real.detach().cpu().numpy()
        #label_feat = label_feat.detach().cpu().numpy()
        input_data = torch.cat((input_data2, z_real, input_data3, known_syn_feat, gsm_syn_feat), dim=0).detach().cpu().numpy()
        #input_data = torch.cat((input_data2, z_real, input_data3, known_syn_feat), dim=0).detach().cpu().numpy()
        label_feat = np.asarray(gt_label[bool_loc].tolist() #['unseen' for i in range(len(input_data2))] 
                                + ['synthetic_unseen' for i in range(len(z_real))] 
                                + ['ood' for i in range(len(input_data3))]
                                + ['cvae_syn' for i in range(len(known_syn_feat))]
                                + ['gsm_syn' for i in range(len(gsm_syn_feat))])
        #tsne_plot(input_data, label_feat, 'unseen_synthetic_cvae')
        #input_data.detach().cpu().numpy()

        ###################
        # train_classifier
        arch = {'num_classes': 5,
                'model': [nn.Linear(z_real.size(1), 1024), nn.ReLU(), nn.Linear(1024, 5)],
                'optim': {'lr': 0.001, 'weight_decay': 0.008}}
        
        # arch = {'model': [nn.Linear(z_real.size(1), 5)],
        #         'optim': {'lr': 0.001, 'weight_decay': 0.001}}

        # print("z_real_shape: ", z_real.detach().cpu().numpy().shape)
        # print("svae_label_shape: ", svae_label.detach().cpu().numpy().shape)
        
        # print("gsm_syn_label: ", np.unique(gsm_syn_label.detach().cpu().numpy()))
        print("gsm_syn_feat: ", gsm_syn_feat.size())
        print("gsm_syn_label: ", gsm_syn_label.size())
        # sys.exit()
        
        # Dataset to train the unseen classifier
        # SVAE
        # [z_real.detach().cpu().numpy(), svae_label.detach().cpu().numpy()]
        # CVAE
        # [known_syn_feat.detach().cpu().numpy(), known_syn_label.detach().cpu().numpy()]
        # GSM-Flow
        # [gsm_syn_feat.detach().cpu().numpy(), gsm_syn_label.detach().cpu().numpy()]
        arch = {'num_classes': 5,
                'model': [nn.Linear(z_real.size(1), 5)],
                'optim': {'lr': 0.001, 'betas': (0.5, 0.999)}} # 'weight_decay': 0.008
        scaler = preprocessing.MinMaxScaler()
        
        train_data_cl5 = gsm_syn_feat.detach().cpu().numpy()
        train_label_cl5 = gsm_syn_label.detach().cpu().numpy()
        classes_unseen = [0,1,6,7,9]
        bool_gsm = np.asarray([True if train_label_cl5[i] in classes_unseen else False for i in range(len(train_label_cl5))])
        train_label_cl5 = train_label_cl5[bool_gsm]
        train_label_cl5 = np.asarray([classes_unseen.index(train_label_cl5[i]) for i in range(len(train_label_cl5))])
        classifier([train_data_cl5[bool_gsm], train_label_cl5], 
                   [input_data2.detach().cpu().numpy(), gt_label[bool_loc]-40],#[test_data_cl5, gt_label[bool_loc]-40],  #[checkpoint_gsm['test_feat'], checkpoint_gsm['test_label']]
                   arch=arch, n_epochs=30, batch_size=1200)
        
        
        # input_data = torch.cat((input_data2, input_data3, known_syn_feat), dim=0).detach().cpu().numpy()
        # input_data = normalize(input_data)
        # label_feat = np.asarray(gt_label[bool_loc].tolist() #['unseen' for i in range(len(input_data2))] 
        #                         + ['ood' for i in range(len(input_data3))]
        #                         + ['cvae_syn' for i in range(len(known_syn_feat))])
        # tsne_plot(input_data, label_feat, 'cvae_syn_unseen')
        
        
        
    def testing_split_ood_synthetic(self, dataset_config, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        all_class = dataset_config["seen_classes"] + dataset_config["unseen_classes"]
        all_seen_class = dataset_config["seen_classes"]
        all_unseen_class = dataset_config["unseen_classes"]
        
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = dataset_config["dataset_name"]
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        seen_train_feat = scaler.fit_transform(seen_train_feat)
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
        
        
        
        
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/{}_data_seen.pt'.format(dataset_config["dataset_name"]))
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/{}_data_unseen.pt'.format(dataset_config["dataset_name"]))
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        if dataset_config["dataset_name"] == 'FLO':
            seen_feat = np.array(seen_feat1)
            seen_label = np.array(seen_label1) + 1
        else:
            seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
            seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1

        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + all_seen_class
            else:
                seen_label[i] = all_class

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        ######################
        ##unseen_feat = normalize(unseen_feat)
        unseen_feat = scaler.transform(unseen_feat)
        unseen_feat = torch.from_numpy(unseen_feat).float()
        mx = unseen_feat.max()
        unseen_feat.mul_(1 / mx)
        unseen_feat = unseen_feat.numpy()
        #######################
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + all_seen_class
            else:
                unseen_label[i] = all_class

        gt_feat = np.concatenate((seen_feat, unseen_feat),  axis=0)
        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        #seen_anchors = all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        unseen_anchors = all_anchors    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(unseen_feat), torch.from_numpy(unseen_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        for i_batch, (input_data, input_label) in enumerate(test_loader):
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                # input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]
                gt.append(kk.data.item())
                z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, unseen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if dist.max()<thresholds[max_idx]: 
                    if len(ood_feat_split) == 0:
                        ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        # print("ood_label_gt ", ood_label_gt)
                        # print("input_label ", input_label[k].detach().cpu().numpy())
                        ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
                    unseen_count +=1
                else:
                    if len(unseen_feat_split) == 0:
                        unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
                    
        #print('ood_label_gt: ', ood_label_gt)
        result = {}
        total_ood = len(gt_label[gt_label == all_class])
        ood_split = len(ood_label_gt[ood_label_gt == all_class])
        print("ood_percentage: ", ood_split/total_ood)
        result['ood_acc'] = ood_split/total_ood
        result['ood_top1'] = ood_split/total_ood
        #####################################
        full_unseen_test_loc = gt_label == all_class
        full_unseen_test_feat = gt_feat[full_unseen_test_loc]
        full_unseen_test_target = gt_label[full_unseen_test_loc] - all_seen_class
        additional_y = []
        additional_pred = []
        idx_actuall = []
        x = np.array(ood_feat_split)
        for i in range(len(full_unseen_test_target)):
            idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
            if len(idx) == 0:
                additional_y.append(full_unseen_test_target[i])
                additional_pred.append(-1)
            else:
                idx_actuall.append(idx[0])
        additional_y = np.array(additional_y)
        additional_pred = np.array(additional_pred)
        idx_actuall = np.array(idx_actuall, dtype=np.int64)
        assert len(idx_actuall) <= len(full_unseen_test_target)
        #####################################
        y = np.array(ood_label_gt - all_seen_class)
        y = np.concatenate((y[idx_actuall], additional_y), axis=0)
        pred = np.array([all_unseen_class for i in range(len(ood_label_gt))])
        pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
        print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
        assert len(pred) == len(full_unseen_test_target)
        F1_score = f1_score(y, pred, average=None, labels=[all_unseen_class])
        print('F1_score_ood: ', np.average(F1_score))
        result['ood_F1'] = np.average(F1_score)
        print()

        
        try:
            total_unseen = len(gt_label[np.logical_and(gt_label < all_class, gt_label >= all_seen_class)])
            unseen_split = len(unseen_label_gt[np.logical_and(unseen_label_gt < all_class, unseen_label_gt >= all_seen_class)])
            print("unseen_percentage: ", unseen_split/total_unseen)

            m, s = self.encoder(torch.from_numpy(unseen_feat_split).cuda())   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            pred = torch.argmax(self.classifier(z_real), dim=1).detach().cpu().numpy()
            acc = np.sum(pred==(unseen_label_gt-all_seen_class))
            print("unseen_acc:", acc/total_unseen)
            result['unseen_acc'] = acc/total_unseen
            acc_top1 = top1_accuracy(pred, unseen_label_gt - all_seen_class, gt_label[np.logical_and(gt_label < all_class, gt_label >= all_seen_class)] - all_seen_class)
            print('acc_unseen_top1: ', acc_top1)
            result['unseen_top1'] = acc_top1
            #####################################
            full_unseen_test_loc = np.logical_and(gt_label >= all_seen_class, gt_label < all_class)
            full_unseen_test_feat = gt_feat[full_unseen_test_loc]
            full_unseen_test_target = gt_label[full_unseen_test_loc] - all_seen_class
            additional_y = []
            additional_pred = []
            idx_actuall = []
            x = np.array(unseen_feat_split)
            for i in range(len(full_unseen_test_target)):
                idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
                if len(idx) == 0:
                    additional_y.append(full_unseen_test_target[i])
                    additional_pred.append(-1)
                else:
                    idx_actuall.append(idx[0])
            additional_y = np.array(additional_y)
            additional_pred = np.array(additional_pred)
            idx_actuall = np.array(idx_actuall, dtype=np.int64)
            assert len(idx_actuall) <= len(full_unseen_test_target)
            #####################################
            y = np.array(unseen_label_gt - all_seen_class)
            y = np.concatenate((y[idx_actuall], additional_y), axis=0)
            pred = np.array(pred)
            pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
            print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
            assert len(pred) == len(full_unseen_test_target)
            F1_score = f1_score(y, pred, average=None, labels=range(0,all_unseen_class))
            print('F1_score_unseen: ', np.average(F1_score))
            result['unseen_F1'] = np.average(F1_score)
            print()
        except:
            print("unseen_acc:", None)
            result['unseen_acc'] = None
            print('acc_unseen_top1: ', None)
            result['unseen_top1'] = None
            print('F1_score_unseen: ', None)
            result['unseen_F1'] = None

        return result
        
        
        
        # Visualize:
        # m, s = self.encoder(torch.from_numpy(unseen_feat).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()
        # input_data = z_real.detach().cpu().numpy() #torch.cat((input_data2, input_data3, known_syn_feat), dim=0).detach().cpu().numpy()
        # #input_data = normalize(input_data)
        # label_feat = np.array(unseen_label)
        # label_feat[label_feat < 40] = -1
        # print('len: 41 ', len(input_data[label_feat == 41]))
        # tsne_plot(input_data[label_feat == 41], label_feat[label_feat == 41], 'svae_stage_2')
        
        
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))        
        

    def testing_split_ood_synthetic_val(self, dataset_config, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        all_class = dataset_config["seen_classes"] + dataset_config["unseen_classes"]
        all_seen_class = dataset_config["seen_classes"]
        all_unseen_class = dataset_config["unseen_classes"]
        
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = dataset_config["dataset_name"]
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        # seen_train_feat = feats[train_idx]
        # seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        # seen_train_feat = scaler.fit_transform(seen_train_feat)
        # seen_train_feat = torch.from_numpy(seen_train_feat).float()
        # mx = seen_train_feat.max()
        # seen_train_feat.mul_(1 / mx)
        # seen_train_feat = seen_train_feat.numpy()

        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        _seen_train_feat = scaler.fit_transform(seen_train_feat)
        _seen_train_feat = torch.from_numpy(_seen_train_feat).float()
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = _seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
        
        
        
        
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/{}_data_seen.pt'.format(dataset_config["dataset_name"]))
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/{}_data_unseen.pt'.format(dataset_config["dataset_name"]))
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        if dataset_config["dataset_name"] == 'FLO':
            seen_feat = np.array(seen_feat1)
            seen_label = np.array(seen_label1) + 1
        else:
            seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
            seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1

        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + all_seen_class
            else:
                seen_label[i] = all_class

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        ######################
        ##unseen_feat = normalize(unseen_feat)
        unseen_feat = scaler.transform(unseen_feat)
        unseen_feat = torch.from_numpy(unseen_feat).float()
        mx = unseen_feat.max()
        unseen_feat.mul_(1 / mx)
        unseen_feat = unseen_feat.numpy()
        #######################
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + all_seen_class
            else:
                unseen_label[i] = all_class

        gt_feat = np.concatenate((seen_feat, unseen_feat),  axis=0)
        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        #seen_anchors = all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        unseen_anchors = all_anchors    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(unseen_feat), torch.from_numpy(unseen_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        for i_batch, (input_data, input_label) in enumerate(test_loader):
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                # input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]
                gt.append(kk.data.item())
                z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, unseen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if dist.max()<thresholds[max_idx]: 
                    if len(ood_feat_split) == 0:
                        ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        # print("ood_label_gt ", ood_label_gt)
                        # print("input_label ", input_label[k].detach().cpu().numpy())
                        ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
                    unseen_count +=1
                else:
                    if len(unseen_feat_split) == 0:
                        unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
                    
        try:
            #print('ood_label_gt: ', ood_label_gt)
            result = {}
            total_ood = len(gt_label[gt_label == all_class])
            ood_split = len(ood_label_gt[ood_label_gt == all_class])
            print("ood_percentage: ", ood_split/total_ood)
            result['ood_acc'] = ood_split/total_ood
            result['ood_top1'] = ood_split/total_ood
            #####################################
            full_unseen_test_loc = gt_label == all_class
            full_unseen_test_feat = gt_feat[full_unseen_test_loc]
            full_unseen_test_target = gt_label[full_unseen_test_loc] - all_seen_class
            additional_y = []
            additional_pred = []
            idx_actuall = []
            x = np.array(ood_feat_split)
            for i in range(len(full_unseen_test_target)):
                idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
                if len(idx) == 0:
                    additional_y.append(full_unseen_test_target[i])
                    additional_pred.append(-1)
                else:
                    idx_actuall.append(idx[0])
            additional_y = np.array(additional_y)
            additional_pred = np.array(additional_pred)
            idx_actuall = np.array(idx_actuall, dtype=np.int64)
            assert len(idx_actuall) <= len(full_unseen_test_target)
            #####################################
            y = np.array(ood_label_gt - all_seen_class)
            y = np.concatenate((y[idx_actuall], additional_y), axis=0)
            pred = np.array([all_unseen_class for i in range(len(ood_label_gt))])
            pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
            print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
            assert len(pred) == len(full_unseen_test_target)
            F1_score = f1_score(y, pred, average=None, labels=[all_unseen_class])
            print('F1_score_ood: ', np.average(F1_score))
            result['ood_F1'] = np.average(F1_score)
            print()
        except:
            result['ood_acc'] = None
            print('ood_acc', result['ood_acc'])
            result['ood_top1'] = None
            print('ood_top1', result['ood_top1'])
            result['ood_F1'] = None
            print('ood_F1', result['ood_F1'])

        try:
            total_unseen = len(gt_label[np.logical_and(gt_label < all_class, gt_label >= all_seen_class)])
            unseen_split = len(unseen_label_gt[np.logical_and(unseen_label_gt < all_class, unseen_label_gt >= all_seen_class)])
            print("unseen_percentage: ", unseen_split/total_unseen)

            m, s = self.encoder(torch.from_numpy(unseen_feat_split).cuda())   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            pred = torch.argmax(self.classifier(z_real), dim=1).detach().cpu().numpy()
            acc = np.sum(pred==(unseen_label_gt-all_seen_class))
            print("unseen_acc:", acc/total_unseen)
            result['unseen_acc'] = acc/total_unseen
            acc_top1 = top1_accuracy(pred, unseen_label_gt - all_seen_class, gt_label[np.logical_and(gt_label < all_class, gt_label >= all_seen_class)] - all_seen_class)
            print('acc_unseen_top1: ', acc_top1)
            result['unseen_top1'] = acc_top1
            #####################################
            full_unseen_test_loc = np.logical_and(gt_label >= all_seen_class, gt_label < all_class)
            full_unseen_test_feat = gt_feat[full_unseen_test_loc]
            full_unseen_test_target = gt_label[full_unseen_test_loc] - all_seen_class
            additional_y = []
            additional_pred = []
            idx_actuall = []
            x = np.array(unseen_feat_split)
            for i in range(len(full_unseen_test_target)):
                idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
                if len(idx) == 0:
                    additional_y.append(full_unseen_test_target[i])
                    additional_pred.append(-1)
                else:
                    idx_actuall.append(idx[0])
            additional_y = np.array(additional_y)
            additional_pred = np.array(additional_pred)
            idx_actuall = np.array(idx_actuall, dtype=np.int64)
            assert len(idx_actuall) <= len(full_unseen_test_target)
            #####################################
            y = np.array(unseen_label_gt - all_seen_class)
            y = np.concatenate((y[idx_actuall], additional_y), axis=0)
            pred = np.array(pred)
            pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
            print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
            assert len(pred) == len(full_unseen_test_target)
            F1_score = f1_score(y, pred, average=None, labels=range(0,all_unseen_class))
            print('F1_score_unseen: ', np.average(F1_score))
            result['unseen_F1'] = np.average(F1_score)
            print()
        except:
            print("unseen_acc:", None)
            result['unseen_acc'] = None
            print('acc_unseen_top1: ', None)
            result['unseen_top1'] = None
            print('F1_score_unseen: ', None)
            result['unseen_F1'] = None

        
        ################################################################
        # Seen As OOD:
        seen_ood_count = 0
        
        m, s = self.encoder(torch.from_numpy(seen_train_feat).cuda())   
        z_real = self.reparameterize(m, s).rsample().squeeze()
        z_real = m.squeeze()
        
        for k in range(z_real.shape[0]):
            z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
            dist = F.cosine_similarity(z_tile, unseen_anchors)
            max_idx = torch.argmax(dist)
            mean_dist += dist.max()
            dist_list.append(dist.max().item())
            all_count += 1  
            '''
            if kk.item() in unseen_labels.tolist():
                unseen_all +=1
                if dist.max()<thresholds[max_idx]: 
                    unseen_count +=1
            elif kk.item() in seen_labels.tolist():
                seen_all +=1  
                if dist.max()>=thresholds[max_idx]:
                    seen_count +=1    
            
            '''
            if dist.max()<thresholds[max_idx]: 
                seen_ood_count += 1
                
                
        total_ood = len(seen_train_feat)
        ood_split = seen_ood_count
        print("seen_ood_percentage: ", ood_split/total_ood)
        result['seen_ood_acc'] = ood_split/total_ood
        result['seen_ood_top1'] = ood_split/total_ood        
        
        
        return result
        
        
        
        # Visualize:
        # m, s = self.encoder(torch.from_numpy(unseen_feat).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()
        # input_data = z_real.detach().cpu().numpy() #torch.cat((input_data2, input_data3, known_syn_feat), dim=0).detach().cpu().numpy()
        # #input_data = normalize(input_data)
        # label_feat = np.array(unseen_label)
        # label_feat[label_feat < 40] = -1
        # print('len: 41 ', len(input_data[label_feat == 41]))
        # tsne_plot(input_data[label_feat == 41], label_feat[label_feat == 41], 'svae_stage_2')
        
        
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))        
        

        
        
        
        
    def testing_split_ood_synthetic_onego(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = 'AWA2'
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        seen_train_feat = scaler.fit_transform(seen_train_feat)
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
        
        
        
        
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_unseen.pt')
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
        ######################
        seen_feat = scaler.transform(seen_feat)
        seen_feat = torch.from_numpy(seen_feat).float()
        mx = seen_feat.max()
        seen_feat.mul_(1 / mx)
        seen_feat = seen_feat.numpy()
        #######################
        seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1
        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
            else:
                seen_label[i] = 45

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        ######################
        unseen_feat = scaler.transform(unseen_feat)
        unseen_feat = torch.from_numpy(unseen_feat).float()
        mx = unseen_feat.max()
        unseen_feat.mul_(1 / mx)
        unseen_feat = unseen_feat.numpy()
        #######################
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
            else:
                unseen_label[i] = 45

        gt_feat = np.concatenate((seen_feat, unseen_feat), axis=0)
        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs[0:40]).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0] + unseen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        #seen_anchors = all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        unseen_anchors = all_anchors    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(gt_feat), torch.from_numpy(gt_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        for i_batch, (input_data, input_label) in enumerate(test_loader):
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                # input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]
                gt.append(kk.data.item())
                z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, unseen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if dist.max()<thresholds[max_idx]: 
                    if len(ood_feat_split) == 0:
                        ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        # print("ood_label_gt ", ood_label_gt)
                        # print("input_label ", input_label[k].detach().cpu().numpy())
                        ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
                    unseen_count +=1
                else:
                    if len(unseen_feat_split) == 0:
                        unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
                    
        print('ood_label_gt: ', ood_label_gt)
        
        total_ood = len(gt_label[gt_label == 45])
        ood_split = len(ood_label_gt[ood_label_gt == 45])
        print("ood_percentage: ", ood_split/total_ood)

        total_unseen = len(gt_label[np.logical_and(gt_label < 45, gt_label >= 40)])
        unseen_split = len(ood_label_gt[np.logical_and(ood_label_gt < 45, ood_label_gt >= 40)])
        print("unseen_percentage: ", unseen_split/total_unseen)

        total_seen = len(gt_label[gt_label < 40])
        seen_split = len(unseen_label_gt[unseen_label_gt < 40])
        print("seen_percentage: ", seen_split/total_seen)

       
        # m, s = self.encoder(torch.from_numpy(unseen_feat_split).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()

        # pred = torch.argmax(self.classifier(z_real), dim=1).detach().cpu().numpy()
        # pred_unseen = pred >= 40
        # target_unseen = unseen_label_gt >= 40
        # unseen_len = np.sum(pred_unseen * target_unseen)
        # print('unseen_len: ', unseen_len)

        # total_ood = len(gt_label[gt_label >= 40])
        # ood_split = len(ood_label_gt[ood_label_gt >= 40]) + unseen_len #+ len(unseen_label_gt[unseen_label_gt >= 40])
        # print("rest_percentage: ", ood_split/total_ood)
        # print("unique_labels_ood: ", np.unique(ood_label_gt))
        # print("unique_labels_unseen: ", np.unique(unseen_label_gt))

    
        # # Visualize:
        # m, s = self.encoder(torch.from_numpy(gt_feat).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()
        # input_data = z_real.detach().cpu().numpy() #torch.cat((input_data2, input_data3, known_syn_feat), dim=0).detach().cpu().numpy()
        # #input_data = normalize(input_data)
        # label_feat = np.array(gt_label)
        # #label_feat[label_feat < 40] = -1
        # #print('len: 41 ', len(input_data[label_feat == 41]))
        # tsne_plot(input_data, label_feat, 'svae_onego')
        
        
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))        
        

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # test_dataset = TensorDataset(torch.from_numpy(unseen_feat), torch.from_numpy(unseen_label))
        # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        # for i_batch, (input_data, input_label) in enumerate(test_loader):
        #     if torch.cuda.is_available():
        #         input_data = input_data.float().cuda()
        #         input_label = input_label.cuda()  
        #         # input_attr = input_attr.float().cuda()  
                                
        #     m, s = self.encoder(input_data)   
        #     z_real = self.reparameterize(m, s).rsample().squeeze()
        #     z_real = m.squeeze()
            
        #     for k in range(z_real.shape[0]):
        #         input_k = input_data[k,:]
        #         kk = input_label[k]
        #         gt.append(kk.data.item())
        #         z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
        #         dist = F.cosine_similarity(z_tile, unseen_anchors)
        #         max_idx = torch.argmax(dist)
        #         mean_dist += dist.max()
        #         dist_list.append(dist.max().item())
        #         all_count += 1  
        #         '''
        #         if kk.item() in unseen_labels.tolist():
        #             unseen_all +=1
        #             if dist.max()<thresholds[max_idx]: 
        #                 unseen_count +=1
        #         elif kk.item() in seen_labels.tolist():
        #             seen_all +=1  
        #             if dist.max()>=thresholds[max_idx]:
        #                 seen_count +=1    
                
        #         '''
        #         if dist.max()<thresholds[max_idx]: 
        #             if len(ood_feat_split) == 0:
        #                 ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
        #                 ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

        #             else:
        #                 ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
        #                 # print("ood_label_gt ", ood_label_gt)
        #                 # print("input_label ", input_label[k].detach().cpu().numpy())
        #                 ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
        #             unseen_count +=1
        #         else:
        #             if len(unseen_feat_split) == 0:
        #                 unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
        #                 unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

        #             else:
        #                 unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
        #                 unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
                    
        # total_ood = len(gt_label[gt_label == 45])
        # ood_split = len(ood_label_gt[ood_label_gt == 45])
        # print("ood_percentage: ", ood_split/total_ood)

        # total_unseen = len(gt_label[np.logical_and(gt_label < 45, gt_label >= 40)])
        # unseen_split = len(unseen_label_gt[np.logical_and(unseen_label_gt < 45, unseen_label_gt >= 40)])
        # print("unseen_percentage: ", unseen_split/total_unseen)
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))
        
        
        
    def testing_seen(self, epoch, seen_classes, known_unseen_classes, classifier2, dataset = None, threshold = 0.99):
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = 'AWA2'
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        seen_train_feat = scaler.fit_transform(seen_train_feat)
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
        
        
        
        
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_unseen.pt')
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
        seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1
        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
            else:
                seen_label[i] = 45

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        ######################
        # unseen_feat = scaler.transform(unseen_feat)
        # unseen_feat = torch.from_numpy(unseen_feat).float()
        # mx = unseen_feat.max()
        # unseen_feat.mul_(1 / mx)
        # unseen_feat = unseen_feat.numpy()
        #######################
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
            else:
                unseen_label[i] = 45

        gt_feat = np.concatenate((seen_feat, unseen_feat), axis=0)
        ######################
        gt_feat = torch.from_numpy(gt_feat).float()
        mx = gt_feat.max()
        gt_feat.mul_(1 / mx)
        gt_feat = gt_feat.numpy()
        #######################
        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        
        # Training classifier2:
        classifier2 = classifier2.cuda()
        optim_cl2 = torch.optim.Adam(classifier2.parameters(), 1e-4)
        
        seen_data = np.array(self.data.train_set)
        seen_data_label = np.array(self.data.train_labels) - 1
        for i in range(len(seen_data_label)):
            seen_data_label[i] = seen_classes.index(seen_data_label[i,0])
            
        seen_dataset = TensorDataset(torch.from_numpy(seen_data).float(), torch.from_numpy(seen_data_label).long())
        seen_dataloader = DataLoader(seen_dataset, batch_size=128, shuffle=True)

        unseen_data = np.array(self.data.Unseen_Data)
        unseen_label = np.array(self.data.Unseen_Labels)
        for i in range(len(unseen_label)):
            unseen_label[i] = 40
        unseen_dataset = TensorDataset(torch.from_numpy(unseen_data).float(), torch.from_numpy(unseen_label).long())
        unseen_dataloader = DataLoader(unseen_dataset, batch_size=128, shuffle=True)
        
        
        for epoch in range(10):
            loss_list = []
            for (x, y) in seen_dataloader:
                x, y = x.cuda(), y.cuda().view(-1)
                ood_iter = iter(unseen_dataloader)
                x_ood, y_ood = next(ood_iter)
                x_ood, y_ood = x_ood.cuda(), y_ood.cuda()
                #x, y = torch.cat((x, x_ood), dim=0), torch.cat((y, y_ood), dim=0)

                m, s = self.encoder(x)   
                z_real = self.reparameterize(m, s).rsample().squeeze()
                z_real = m.squeeze()

                m_ood, s_ood = self.encoder(x_ood)   
                z_real_ood = self.reparameterize(m_ood, s_ood).rsample().squeeze()
                z_real_ood = m_ood.squeeze()

                pred1 = classifier2(z_real)
                pred2 = classifier2(z_real_ood)
                loss = F.nll_loss(pred1, y) + F.nll_loss(pred2, y_ood)
                loss_list.append(loss.item())
                optim_cl2.zero_grad()
                loss.backward()
                optim_cl2.step()
            
            # Testing:
            with torch.no_grad():
                m, s = self.encoder(torch.from_numpy(gt_feat[gt_label<40]).cuda())   
                z_real = self.reparameterize(m, s).rsample().squeeze()
                z_real = m.squeeze()
                pred = classifier2(z_real)
                pred = torch.argmax(pred, dim=1).detach().cpu().numpy()
                target = gt_label[gt_label<40]
                acc = np.sum(pred == target)/len(z_real)
                print("seen:   Epoch: {}  :: Acc: {} :: mean: {}".format(epoch, acc, np.mean(loss_list)))
                
                m, s = self.encoder(torch.from_numpy(gt_feat[gt_label>=40]).cuda())   
                z_real = self.reparameterize(m, s).rsample().squeeze()
                z_real = m.squeeze()
                pred = classifier2(z_real)
                pred = torch.argmax(pred, dim=1).detach().cpu().numpy()
                target = gt_label[gt_label>=40]
                acc = np.sum(pred == target)/len(z_real)
                print("unseen: Epoch: {}  :: Acc: {} :: mean: {}".format(epoch, acc, np.mean(loss_list)))

                m, s = self.encoder(torch.from_numpy(unseen_data).cuda())   
                z_real = self.reparameterize(m, s).rsample().squeeze()
                z_real = m.squeeze()
                pred = classifier2(z_real)
                pred = torch.argmax(pred, dim=1).detach().cpu().numpy()
                target = unseen_label
                acc = np.sum(pred == target)/len(z_real)
                print("unseen(train): Epoch: {}  :: Acc: {} :: mean: {}".format(epoch, acc, np.mean(loss_list)))

        
        
        # m, s = self.encoder(torch.from_numpy(seen_feat).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()
        # pred = self.classifier(z_real)
        # pred = torch.argmax(pred, dim=1).detach().cpu().numpy()
        # for i in range(len(pred)):
        #     if pred[i] in seen_classes:
        #         pred[i] = seen_classes.index(pred[i])
        #     elif pred[i] in known_unseen_classes:
        #         pred[i] = known_unseen_classes.index(pred[i]) + 40
        #     else:
        #         pred[i] = 45
        # acc = np.sum(pred==seen_label)
        # #print("seen_acc:", acc/len(labels[attrs_mat['test_seen_loc'].squeeze() - 1]))
        # #print("seen_percentage: ", len(seen_label[seen_label < 40])/len(gt_label[gt_label<40]))
        # #print("len(seen): ", len(gt_label[gt_label<40]), len(labels[attrs_mat['test_seen_loc'].squeeze() - 1]))
        # #print("len(unseen): ", len(gt_label[gt_label>=40]), len(labels[attrs_mat['test_unseen_loc'].squeeze() - 1]))
        # #print("len(total): ", len(gt_label), len(labels[attrs_mat['test_seen_loc'].squeeze() - 1]) + len(labels[attrs_mat['test_unseen_loc'].squeeze() - 1]))
        # acc_top1 = top1_accuracy(pred, seen_label, gt_label[gt_label < 40])
        # print('acc_seen_top1: ', acc_top1)
        # y = np.array(seen_label)
        # #####################################
        # full_unseen_test_loc = gt_label < 40
        # full_unseen_test_feat = gt_feat[full_unseen_test_loc]
        # full_unseen_test_target = gt_label[full_unseen_test_loc]
        # additional_y = []
        # additional_pred = []
        # idx_actuall = []
        # #assert len(y) < len(full_unseen_test_target)
        # x = np.array(seen_feat)
        # for i in range(len(full_unseen_test_target)):
        #     idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
        #     if len(idx) == 0:
        #         additional_y.append(full_unseen_test_target[i])
        #         additional_pred.append(-1)
        #     else:
        #         idx_actuall.append(idx[0])
        # additional_y = np.array(additional_y)
        # additional_pred = np.array(additional_pred)
        # idx_actuall = np.array(idx_actuall, dtype=np.int64)
        # print(idx_actuall)
        # assert len(idx_actuall) < len(full_unseen_test_target)
        # #####################################
        # y = np.concatenate((y[idx_actuall], additional_y), axis=0)
        # pred = np.array(pred)
        # pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
        # print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
        # assert len(pred) == len(full_unseen_test_target)
        # F1_score = f1_score(y, pred, average=None, labels=range(0, 40))
        # print('F1_score_unseen: ', np.average(F1_score))
        # print()

              
        
        
        # # data_dict = {}
        # # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))        
               


    def testing_split_ood_synthetic_save_data(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = 'AWA2'
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        seen_train_feat = scaler.fit_transform(seen_train_feat)
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
        
        
        
        
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_unseen.pt')
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
        seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1
        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
            else:
                seen_label[i] = 45

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        ######################
        unseen_feat = scaler.transform(unseen_feat)
        unseen_feat = torch.from_numpy(unseen_feat).float()
        mx = unseen_feat.max()
        unseen_feat.mul_(1 / mx)
        unseen_feat = unseen_feat.numpy()
        #######################
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
            else:
                unseen_label[i] = 45

        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        #seen_anchors = all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        unseen_anchors = all_anchors    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(unseen_feat), torch.from_numpy(unseen_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        
        with torch.no_grad():
            input_data, input_label = torch.from_numpy(unseen_feat).cuda(), torch.from_numpy(unseen_label).cuda()
            m, s = self.encoder(input_data)  
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()

            data_dict = {'feat': z_real.detach().cpu().numpy(),
                        'label': input_label.detach().cpu().numpy(),
                        'cls_dict': self.classifier.state_dict()}
        
            train_data, train_label = torch.from_numpy(dataset.train_set), torch.from_numpy(dataset.train_labels - 1)
            train_dataset = TensorDataset(train_data, train_label)
            train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=512)
            for i, (x, y) in enumerate(train_dataloader):
                m, s = self.encoder(x.cuda())  
                z_real = self.reparameterize(m, s).rsample().squeeze()
                z_real = m.squeeze()
                if i == 0:
                    train_feat = z_real.detach().cpu().numpy()
                    train_labels = y.detach().cpu().numpy()
                else:
                    train_feat = np.concatenate((train_feat, z_real.detach().cpu().numpy()), axis=0)
                    train_labels = np.concatenate((train_labels, y.detach().cpu().numpy()), axis=0)

            data_dict['train_feat'] = train_feat
            data_dict['train_label'] = train_labels
            print('train_labels: ', np.unique(dataset.train_labels))
            torch.save(data_dict, '/home/sethupathy/openworl_zsl/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/src/latent_data/latent_data_svae_synthetic_600.zip')
        
       
        
    def testing_split_ood_seen_as_ood(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_unseen.pt')
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
        seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1
        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
            else:
                seen_label[i] = 45

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
            else:
                unseen_label[i] = 45

        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        # seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors #[unseen_idx.tolist(),:]    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(unseen_feat), torch.from_numpy(unseen_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        for i_batch, (input_data, input_label) in enumerate(test_loader):
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                # input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k] - 40
                gt.append(kk.data.item())
                z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, unseen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if dist.max()<thresholds[max_idx]: 
                    if len(ood_feat_split) == 0:
                        ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        # print("ood_label_gt ", ood_label_gt)
                        # print("input_label ", input_label[k].detach().cpu().numpy())
                        ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
                    unseen_count +=1
                else:
                    if len(unseen_feat_split) == 0:
                        unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
                    
        total_ood = len(gt_label[gt_label == 45])
        ood_split = len(ood_label_gt[ood_label_gt == 45])
        print("ood_percentage: ", ood_split/total_ood)

        try:
            total_unseen = len(gt_label[np.logical_and(gt_label < 45, gt_label >= 40)])
            unseen_split = len(unseen_label_gt[np.logical_and(unseen_label_gt < 45, unseen_label_gt >= 40)])
            print("unseen_percentage: ", unseen_split/total_unseen)
        except:
            pass
        
        
        m, s = self.encoder(torch.from_numpy(unseen_feat).cuda())   
        z_real = self.reparameterize(m, s).rsample().squeeze()
        z_real = m.squeeze()

        pred = torch.argmax(self.classifier(z_real), dim=1).detach().cpu().numpy()

        total_ood = len(gt_label[gt_label == 45])
        ood_split = np.sum((pred == 5) * ((unseen_label - 40) == 5))
        print("cls_ood_percentage: ", ood_split/total_ood)

        total_unseen = len(gt_label[np.logical_and(gt_label < 45, gt_label >= 40)])
        unseen_split = len(pred[pred < 5])
        print("cls_unseen_percentage: ", unseen_split/total_unseen)

        
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))
               
               
    def testing_seen_vs_rest(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = 'AWA2'
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        seen_train_feat = scaler.fit_transform(seen_train_feat)
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
        
        
        
        
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_unseen.pt')
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
        ######################
        # seen_feat = scaler.transform(seen_feat)
        # seen_feat = torch.from_numpy(seen_feat).float()
        # mx = seen_feat.max()
        # seen_feat.mul_(1 / mx)
        # seen_feat = seen_feat.numpy()
        #######################
        seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1
        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
            else:
                seen_label[i] = 45

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        ######################
        # unseen_feat = scaler.transform(unseen_feat)
        # unseen_feat = torch.from_numpy(unseen_feat).float()
        # mx = unseen_feat.max()
        # unseen_feat.mul_(1 / mx)
        # unseen_feat = unseen_feat.numpy()
        #######################
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
            else:
                unseen_label[i] = 45

        gt_feat = np.concatenate((seen_feat, unseen_feat), axis=0)
        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs[0:40]).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0] + unseen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        #seen_anchors = all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        unseen_anchors = all_anchors    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(gt_feat), torch.from_numpy(gt_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        for i_batch, (input_data, input_label) in enumerate(test_loader):
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                # input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]
                gt.append(kk.data.item())
                z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, unseen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if dist.max()<thresholds[max_idx]: 
                    if len(ood_feat_split) == 0:
                        ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        # print("ood_label_gt ", ood_label_gt)
                        # print("input_label ", input_label[k].detach().cpu().numpy())
                        ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
                    unseen_count +=1
                else:
                    if len(unseen_feat_split) == 0:
                        unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
        result = {}           
        print('ood_label_gt: ', ood_label_gt)
        
        total_ood = len(gt_label[gt_label == 45])
        ood_split = len(ood_label_gt[ood_label_gt == 45])
        result['ood'] = ood_split/total_ood
        print("ood_percentage: ", ood_split/total_ood)

        total_unseen = len(gt_label[np.logical_and(gt_label < 45, gt_label >= 40)])
        unseen_split = len(ood_label_gt[np.logical_and(ood_label_gt < 45, ood_label_gt >= 40)])
        print("unseen_percentage: ", unseen_split/total_unseen)
        result['unseen'] = unseen_split/total_unseen

        total_seen = len(gt_label[gt_label < 40])
        seen_split = len(unseen_label_gt[unseen_label_gt < 40])
        print("seen_percentage: ", seen_split/total_seen)
        result['seen'] = seen_split/total_seen

        data_dict = {}
        data_dict['seen_feat'] = np.asarray(unseen_feat_split)    
        data_dict['seen_label_gt'] = np.asarray(unseen_label_gt) 
        data_dict['unseen_feat_split'] = np.asarray(ood_feat_split) 
        data_dict['unseen_label_gt'] = np.asarray(ood_label_gt) 
        torch.save(data_dict, '../data_split_svae/domain_split.zip')        
               
        return result

       
        # m, s = self.encoder(torch.from_numpy(unseen_feat_split).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()

        # pred = torch.argmax(self.classifier(z_real), dim=1).detach().cpu().numpy()
        # pred_unseen = pred >= 40
        # target_unseen = unseen_label_gt >= 40
        # unseen_len = np.sum(pred_unseen * target_unseen)
        # print('unseen_len: ', unseen_len)

        # total_ood = len(gt_label[gt_label >= 40])
        # ood_split = len(ood_label_gt[ood_label_gt >= 40]) + unseen_len #+ len(unseen_label_gt[unseen_label_gt >= 40])
        # print("rest_percentage: ", ood_split/total_ood)
        # print("unique_labels_ood: ", np.unique(ood_label_gt))
        # print("unique_labels_unseen: ", np.unique(unseen_label_gt))

    
        # # Visualize:
        # m, s = self.encoder(torch.from_numpy(gt_feat).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()
        # input_data = z_real.detach().cpu().numpy() #torch.cat((input_data2, input_data3, known_syn_feat), dim=0).detach().cpu().numpy()
        # #input_data = normalize(input_data)
        # label_feat = np.array(gt_label)
        # #label_feat[label_feat < 40] = -1
        # #print('len: 41 ', len(input_data[label_feat == 41]))
        # tsne_plot(input_data, label_feat, 'svae_onego')
        
        
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))        
            
    
    def testing_split_ood_synthetic_domain(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = 'AWA2'
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        seen_train_feat = scaler.fit_transform(seen_train_feat)
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
               
        #checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        checkpoint = torch.load('../data_split_svae/domain_split.zip')        
        seen_feat = checkpoint['seen_feat']
        seen_label = checkpoint['seen_label_gt']
                       
        # for i in range(len(seen_label)):
        #     if seen_label[i] in seen_classes:
        #         seen_label[i] = seen_classes.index(seen_label[i])
        #     elif seen_label[i] in known_unseen_classes:
        #         seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
        #     else:
        #         seen_label[i] = 45
        
        unseen_feat = checkpoint['unseen_feat_split']
        unseen_label = checkpoint['unseen_label_gt']
        ######################
        # unseen_feat = normalize(unseen_feat)
        # unseen_feat = scaler.transform(unseen_feat)
        # unseen_feat = torch.from_numpy(unseen_feat).float()
        # mx = unseen_feat.max()
        # unseen_feat.mul_(1 / mx)
        # unseen_feat = unseen_feat.numpy()
        #######################
        # for i in range(len(unseen_label)):
        #     if unseen_label[i] in seen_classes:
        #         unseen_label[i] = seen_classes.index(unseen_label[i])
        #     elif unseen_label[i] in known_unseen_classes:
        #         unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
        #     else:
        #         unseen_label[i] = 45

        gt_feat = np.concatenate((seen_feat, unseen_feat),  axis=0)
        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        #seen_anchors = all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        unseen_anchors = all_anchors    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(unseen_feat), torch.from_numpy(unseen_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        for i_batch, (input_data, input_label) in enumerate(test_loader):
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                # input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]
                gt.append(kk.data.item())
                z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, unseen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if dist.max()<thresholds[max_idx]: 
                    if len(ood_feat_split) == 0:
                        ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        # print("ood_label_gt ", ood_label_gt)
                        # print("input_label ", input_label[k].detach().cpu().numpy())
                        ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
                    unseen_count +=1
                else:
                    if len(unseen_feat_split) == 0:
                        unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
                    
        #print('ood_label_gt: ', ood_label_gt)
        result = {}
        total_ood = len(gt_label[gt_label == 45])
        ood_split = len(ood_label_gt[ood_label_gt == 45])
        print("ood_percentage: ", ood_split/total_ood)
        result['ood_acc'] = ood_split/total_ood
        result['ood_top1'] = ood_split/total_ood
        #####################################
        full_unseen_test_loc = gt_label == 45
        full_unseen_test_feat = gt_feat[full_unseen_test_loc]
        full_unseen_test_target = gt_label[full_unseen_test_loc] - 40
        additional_y = []
        additional_pred = []
        idx_actuall = []
        x = np.array(ood_feat_split)
        for i in range(len(full_unseen_test_target)):
            idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
            if len(idx) == 0:
                additional_y.append(full_unseen_test_target[i])
                additional_pred.append(-1)
            else:
                idx_actuall.append(idx[0])
        additional_y = np.array(additional_y)
        additional_pred = np.array(additional_pred)
        idx_actuall = np.array(idx_actuall, dtype=np.int64)
        assert len(idx_actuall) < len(full_unseen_test_target)
        #####################################
        y = np.array(ood_label_gt - 40)
        y = np.concatenate((y[idx_actuall], additional_y), axis=0)
        pred = np.array([5 for i in range(len(ood_label_gt))])
        pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
        print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
        assert len(pred) == len(full_unseen_test_target)
        F1_score = f1_score(y, pred, average=None, labels=[5])
        print('F1_score_ood: ', np.average(F1_score))
        result['ood_F1'] = np.average(F1_score)
        print()

        
        try:
            total_unseen = len(gt_label[np.logical_and(gt_label < 45, gt_label >= 40)])
            unseen_split = len(unseen_label_gt[np.logical_and(unseen_label_gt < 45, unseen_label_gt >= 40)])
            print("unseen_percentage: ", unseen_split/total_unseen)

            m, s = self.encoder(torch.from_numpy(unseen_feat_split).cuda())   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            pred = torch.argmax(self.classifier(z_real), dim=1).detach().cpu().numpy()
            acc = np.sum(pred==(unseen_label_gt-40))
            print("unseen_acc:", acc/total_unseen)
            result['unseen_acc'] = acc/total_unseen
            acc_top1 = top1_accuracy(pred, unseen_label_gt - 40, gt_label[np.logical_and(gt_label < 45, gt_label >= 40)] - 40)
            print('acc_unseen_top1: ', acc_top1)
            result['unseen_top1'] = acc_top1
            #####################################
            full_unseen_test_loc = np.logical_and(gt_label >= 40, gt_label < 45)
            full_unseen_test_feat = gt_feat[full_unseen_test_loc]
            full_unseen_test_target = gt_label[full_unseen_test_loc] - 40
            additional_y = []
            additional_pred = []
            idx_actuall = []
            x = np.array(unseen_feat_split)
            for i in range(len(full_unseen_test_target)):
                idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
                if len(idx) == 0:
                    additional_y.append(full_unseen_test_target[i])
                    additional_pred.append(-1)
                else:
                    idx_actuall.append(idx[0])
            additional_y = np.array(additional_y)
            additional_pred = np.array(additional_pred)
            idx_actuall = np.array(idx_actuall, dtype=np.int64)
            assert len(idx_actuall) < len(full_unseen_test_target)
            #####################################
            y = np.array(unseen_label_gt - 40)
            y = np.concatenate((y[idx_actuall], additional_y), axis=0)
            pred = np.array(pred)
            pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
            print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
            assert len(pred) == len(full_unseen_test_target)
            F1_score = f1_score(y, pred, average=None, labels=range(0,5))
            print('F1_score_unseen: ', np.average(F1_score))
            result['unseen_F1'] = np.average(F1_score)
            print()
        except:
            print("unseen_acc:", None)
            result['unseen_acc'] = None
            print('acc_unseen_top1: ', None)
            result['unseen_top1'] = None
            print('F1_score_unseen: ', None)
            result['unseen_F1'] = None

        return result
           
    def testing_seen_domain(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = 'AWA2'
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        seen_train_feat = scaler.fit_transform(seen_train_feat)
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
        
        
        
        
        #checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        checkpoint = torch.load('../data_split_svae/domain_split.zip')        
        seen_feat = checkpoint['seen_feat']
        seen_label = checkpoint['seen_label_gt']

        unseen_feat = checkpoint['unseen_feat_split']
        unseen_label = checkpoint['unseen_label_gt']

        gt_feat = np.concatenate((seen_feat, unseen_feat), axis=0)
        ######################
        # gt_feat = torch.from_numpy(gt_feat).float()
        # mx = gt_feat.max()
        # gt_feat.mul_(1 / mx)
        # gt_feat = gt_feat.numpy()
        #######################
        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        
        m, s = self.encoder(torch.from_numpy(seen_feat).cuda())   
        z_real = self.reparameterize(m, s).rsample().squeeze()
        z_real = m.squeeze()
        pred = self.classifier(z_real)
        pred = torch.argmax(pred, dim=1).detach().cpu().numpy()
        for i in range(len(pred)):
            if pred[i] in seen_classes:
                pred[i] = seen_classes.index(pred[i])
            elif pred[i] in known_unseen_classes:
                pred[i] = known_unseen_classes.index(pred[i]) + 40
            else:
                pred[i] = 45
        acc = np.sum(pred==seen_label)
        print("seen_acc:", acc/len(labels[attrs_mat['test_seen_loc'].squeeze() - 1]))
        print("seen_percentage: ", len(seen_label[seen_label < 40])/len(gt_label[gt_label<40]))
        print("len(seen): ", len(gt_label[gt_label<40]), len(labels[attrs_mat['test_seen_loc'].squeeze() - 1]))
        print("len(unseen): ", len(gt_label[gt_label>=40]), len(labels[attrs_mat['test_unseen_loc'].squeeze() - 1]))
        print("len(total): ", len(gt_label), len(labels[attrs_mat['test_seen_loc'].squeeze() - 1]) + len(labels[attrs_mat['test_unseen_loc'].squeeze() - 1]))
        acc_top1 = top1_accuracy(pred, seen_label, gt_label[gt_label < 40])
        print('acc_seen_top1: ', acc_top1)
        y = np.array(seen_label)
        #####################################
        full_unseen_test_loc = gt_label < 40
        full_unseen_test_feat = gt_feat[full_unseen_test_loc]
        full_unseen_test_target = gt_label[full_unseen_test_loc]
        additional_y = []
        additional_pred = []
        idx_actuall = []
        #assert len(y) < len(full_unseen_test_target)
        x = np.array(seen_feat)
        for i in range(len(full_unseen_test_target)):
            idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
            if len(idx) == 0:
                additional_y.append(full_unseen_test_target[i])
                additional_pred.append(-1)
            else:
                idx_actuall.append(idx[0])
        additional_y = np.array(additional_y)
        additional_pred = np.array(additional_pred)
        idx_actuall = np.array(idx_actuall, dtype=np.int64)
        # print(idx_actuall)
        assert len(idx_actuall) < len(full_unseen_test_target)
        #####################################
        y = np.concatenate((y[idx_actuall], additional_y), axis=0)
        pred = np.array(pred)
        pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
        print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
        assert len(pred) == len(full_unseen_test_target)
        F1_score = f1_score(y, pred, average=None, labels=range(0, 40))
        print('F1_score_unseen: ', np.average(F1_score))
        print()

              
        
        
        # # data_dict = {}
        # # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))        
     
    
    
    def testing_split_ood_synthetic_cvae(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = 'AWA2'
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        seen_train_feat = scaler.fit_transform(seen_train_feat)
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
        
        
        
        
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        # checkpoint1 = torch.load('../data_split_svae/domain_split.zip')        
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_unseen.pt')
        # checkpoint2 = torch.load('../data_split_svae/domain_split.zip')
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
        seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1
        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
            else:
                seen_label[i] = 45

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        ######################
        #unseen_feat = normalize(unseen_feat)
        # unseen_feat = scaler.transform(unseen_feat)
        # unseen_feat = torch.from_numpy(unseen_feat).float()
        # mx = unseen_feat.max()
        # unseen_feat.mul_(1 / mx)
        # unseen_feat = unseen_feat.numpy()
        #######################
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
            else:
                unseen_label[i] = 45

        gt_feat = np.concatenate((seen_feat, unseen_feat),  axis=0)
        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        #seen_anchors = all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        unseen_anchors = all_anchors    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(unseen_feat), torch.from_numpy(unseen_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        for i_batch, (input_data, input_label) in enumerate(test_loader):
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                # input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]
                gt.append(kk.data.item())
                z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, unseen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if dist.max()<thresholds[max_idx]: 
                    if len(ood_feat_split) == 0:
                        ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        # print("ood_label_gt ", ood_label_gt)
                        # print("input_label ", input_label[k].detach().cpu().numpy())
                        ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
                    unseen_count +=1
                else:
                    if len(unseen_feat_split) == 0:
                        unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
                    
        #print('ood_label_gt: ', ood_label_gt)
        result = {}
        total_ood = len(gt_label[gt_label == 45])
        ood_split = len(ood_label_gt[ood_label_gt == 45])
        print("ood_percentage: ", ood_split/total_ood)
        result['ood_acc'] = ood_split/total_ood
        result['ood_top1'] = ood_split/total_ood
        #####################################
        full_unseen_test_loc = gt_label == 45
        full_unseen_test_feat = gt_feat[full_unseen_test_loc]
        full_unseen_test_target = gt_label[full_unseen_test_loc] - 40
        additional_y = []
        additional_pred = []
        idx_actuall = []
        x = np.array(ood_feat_split)
        for i in range(len(full_unseen_test_target)):
            idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
            if len(idx) == 0:
                additional_y.append(full_unseen_test_target[i])
                additional_pred.append(-1)
            else:
                idx_actuall.append(idx[0])
        additional_y = np.array(additional_y)
        additional_pred = np.array(additional_pred)
        idx_actuall = np.array(idx_actuall, dtype=np.int64)
        assert len(idx_actuall) < len(full_unseen_test_target)
        #####################################
        y = np.array(ood_label_gt - 40)
        y = np.concatenate((y[idx_actuall], additional_y), axis=0)
        pred = np.array([5 for i in range(len(ood_label_gt))])
        pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
        print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
        assert len(pred) == len(full_unseen_test_target)
        F1_score = f1_score(y, pred, average=None, labels=[5])
        print('F1_score_ood: ', np.average(F1_score))
        result['ood_F1'] = np.average(F1_score)
        print()

        
        try:
            total_unseen = len(gt_label[np.logical_and(gt_label < 45, gt_label >= 40)])
            unseen_split = len(unseen_label_gt[np.logical_and(unseen_label_gt < 45, unseen_label_gt >= 40)])
            print("unseen_percentage: ", unseen_split/total_unseen)

            m, s = self.encoder(torch.from_numpy(unseen_feat_split).cuda())   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            pred = torch.argmax(self.classifier(z_real), dim=1).detach().cpu().numpy()
            acc = np.sum(pred==(unseen_label_gt-40))
            print("unseen_acc:", acc/total_unseen)
            result['unseen_acc'] = acc/total_unseen
            acc_top1 = top1_accuracy(pred, unseen_label_gt - 40, gt_label[np.logical_and(gt_label < 45, gt_label >= 40)] - 40)
            print('acc_unseen_top1: ', acc_top1)
            result['unseen_top1'] = acc_top1
            #####################################
            full_unseen_test_loc = np.logical_and(gt_label >= 40, gt_label < 45)
            full_unseen_test_feat = gt_feat[full_unseen_test_loc]
            full_unseen_test_target = gt_label[full_unseen_test_loc] - 40
            additional_y = []
            additional_pred = []
            idx_actuall = []
            x = np.array(unseen_feat_split)
            for i in range(len(full_unseen_test_target)):
                idx = np.flatnonzero((full_unseen_test_feat[i] == x).all(1))
                if len(idx) == 0:
                    additional_y.append(full_unseen_test_target[i])
                    additional_pred.append(-1)
                else:
                    idx_actuall.append(idx[0])
            additional_y = np.array(additional_y)
            additional_pred = np.array(additional_pred)
            idx_actuall = np.array(idx_actuall, dtype=np.int64)
            assert len(idx_actuall) < len(full_unseen_test_target)
            #####################################
            y = np.array(unseen_label_gt - 40)
            y = np.concatenate((y[idx_actuall], additional_y), axis=0)
            pred = np.array(pred)
            pred = np.concatenate((pred[idx_actuall], additional_pred), axis=0)
            print('len of pred and full_unseen_test_target: ', len(pred), len(full_unseen_test_feat))
            assert len(pred) == len(full_unseen_test_target)
            F1_score = f1_score(y, pred, average=None, labels=range(0,5))
            print('F1_score_unseen: ', np.average(F1_score))
            result['unseen_F1'] = np.average(F1_score)
            print()
        except:
            print("unseen_acc:", None)
            result['unseen_acc'] = None
            print('acc_unseen_top1: ', None)
            result['unseen_top1'] = None
            print('F1_score_unseen: ', None)
            result['unseen_F1'] = None

        return result
        
        
        
        # Visualize:
        # m, s = self.encoder(torch.from_numpy(unseen_feat).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()
        # input_data = z_real.detach().cpu().numpy() #torch.cat((input_data2, input_data3, known_syn_feat), dim=0).detach().cpu().numpy()
        # #input_data = normalize(input_data)
        # label_feat = np.array(unseen_label)
        # label_feat[label_feat < 40] = -1
        # print('len: 41 ', len(input_data[label_feat == 41]))
        # tsne_plot(input_data[label_feat == 41], label_feat[label_feat == 41], 'svae_stage_2')
        
        
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))        
        
    
    def testing_seen_vs_rest_cvae_domain(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        scaler = preprocessing.MinMaxScaler()
        #####################################################################
        DATASET = 'AWA2'
        print(f'<=============== Loading data for {DATASET} ===============>')
        DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
        DATA_DIR = f'../../Datasets/{DATASET}'
        data = io.loadmat(f'{DATA_DIR}/res101.mat')
        attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        
        seen_train_feat = feats[train_idx]
        seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])
        seen_train_feat = scaler.fit_transform(seen_train_feat)
        seen_train_feat = torch.from_numpy(seen_train_feat).float()
        mx = seen_train_feat.max()
        seen_train_feat.mul_(1 / mx)
        seen_train_feat = seen_train_feat.numpy()
        #######################################################################
        
        
        
        
        
        checkpoint1 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_seen.pt')
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_unseen.pt')
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        seen_feat = np.concatenate((seen_feat1, seen_feat2), axis=0)
        ######################
        # seen_feat = scaler.transform(seen_feat)
        # seen_feat = torch.from_numpy(seen_feat).float()
        # mx = seen_feat.max()
        # seen_feat.mul_(1 / mx)
        # seen_feat = seen_feat.numpy()
        #######################
        seen_label = np.concatenate((seen_label1, seen_label2), axis=0) + 1
        for i in range(len(seen_label)):
            if seen_label[i] in seen_classes:
                seen_label[i] = seen_classes.index(seen_label[i])
            elif seen_label[i] in known_unseen_classes:
                seen_label[i] = known_unseen_classes.index(seen_label[i]) + 40
            else:
                seen_label[i] = 45

        unseen_feat = np.concatenate((unseen_feat1, unseen_feat2), axis=0)
        ######################
        # unseen_feat = scaler.transform(unseen_feat)
        # unseen_feat = torch.from_numpy(unseen_feat).float()
        # mx = unseen_feat.max()
        # unseen_feat.mul_(1 / mx)
        # unseen_feat = unseen_feat.numpy()
        #######################
        unseen_label = np.concatenate((unseen_label1, unseen_label2), axis=0) + 1
        for i in range(len(unseen_label)):
            if unseen_label[i] in seen_classes:
                unseen_label[i] = seen_classes.index(unseen_label[i])
            elif unseen_label[i] in known_unseen_classes:
                unseen_label[i] = known_unseen_classes.index(unseen_label[i]) + 40
            else:
                unseen_label[i] = 45

        gt_feat = np.concatenate((seen_feat, unseen_feat), axis=0)
        gt_label = np.concatenate((seen_label, unseen_label), axis=0)
            
        all_attrs = torch.Tensor(dataset.attrs[0:40]).float().cuda()
        seen_labels = np.asarray(seen_classes)
        unseen_labels = np.asarray(known_unseen_classes)
        
        thresholds = np.ones(seen_labels.shape[0] + unseen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]        
        seen_idx = seen_labels
        unseen_idx = unseen_labels
        
        #seen_anchors = all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        unseen_anchors = all_anchors    
            
        unseen_count = 0
        unseen_all = 1
        ood_count = 0
        ood_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []

        unseen_feat_split = []
        unseen_label_gt = []
        ood_feat_split = []
        ood_label_gt = []
        
        test_dataset = TensorDataset(torch.from_numpy(gt_feat), torch.from_numpy(gt_label))
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
        for i_batch, (input_data, input_label) in enumerate(test_loader):
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                # input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]
                gt.append(kk.data.item())
                z_tile = z_real[k,:].repeat(unseen_anchors.shape[0]).view(unseen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, unseen_anchors)
                max_idx = torch.argmax(dist)
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1    
                
                '''
                if dist.max()<thresholds[max_idx]: 
                    if len(ood_feat_split) == 0:
                        ood_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        ood_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        ood_feat_split = np.concatenate((ood_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        # print("ood_label_gt ", ood_label_gt)
                        # print("input_label ", input_label[k].detach().cpu().numpy())
                        ood_label_gt = np.concatenate((ood_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                    
                    unseen_count +=1
                else:
                    if len(unseen_feat_split) == 0:
                        unseen_feat_split = input_k.view(1,-1).detach().cpu().numpy()
                        unseen_label_gt = np.asarray([input_label[k].detach().cpu().numpy()])

                    else:
                        unseen_feat_split = np.concatenate((unseen_feat_split, input_k.view(1,-1).detach().cpu().numpy()), axis=0)
                        unseen_label_gt = np.concatenate((unseen_label_gt, np.asarray([input_label[k].detach().cpu().numpy()])), axis=0)
                
        result = {}           
        print('ood_label_gt: ', ood_label_gt)
        
        total_ood = len(gt_label[gt_label == 45])
        ood_split = len(ood_label_gt[ood_label_gt == 45])
        result['ood'] = ood_split/total_ood
        print("ood_percentage: ", ood_split/total_ood)

        total_unseen = len(gt_label[np.logical_and(gt_label < 45, gt_label >= 40)])
        unseen_split = len(ood_label_gt[np.logical_and(ood_label_gt < 45, ood_label_gt >= 40)])
        print("unseen_percentage: ", unseen_split/total_unseen)
        result['unseen'] = unseen_split/total_unseen

        total_seen = len(gt_label[gt_label < 40])
        seen_split = len(unseen_label_gt[unseen_label_gt < 40])
        print("seen_percentage: ", seen_split/total_seen)
        result['seen'] = seen_split/total_seen

        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(unseen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(unseen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(ood_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(ood_label_gt) 
        # torch.save(data_dict, '../data_split_svae/cvae_domain1_split.zip')        
               
        return result

       
        # m, s = self.encoder(torch.from_numpy(unseen_feat_split).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()

        # pred = torch.argmax(self.classifier(z_real), dim=1).detach().cpu().numpy()
        # pred_unseen = pred >= 40
        # target_unseen = unseen_label_gt >= 40
        # unseen_len = np.sum(pred_unseen * target_unseen)
        # print('unseen_len: ', unseen_len)

        # total_ood = len(gt_label[gt_label >= 40])
        # ood_split = len(ood_label_gt[ood_label_gt >= 40]) + unseen_len #+ len(unseen_label_gt[unseen_label_gt >= 40])
        # print("rest_percentage: ", ood_split/total_ood)
        # print("unique_labels_ood: ", np.unique(ood_label_gt))
        # print("unique_labels_unseen: ", np.unique(unseen_label_gt))

    
        # # Visualize:
        # m, s = self.encoder(torch.from_numpy(gt_feat).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()
        # input_data = z_real.detach().cpu().numpy() #torch.cat((input_data2, input_data3, known_syn_feat), dim=0).detach().cpu().numpy()
        # #input_data = normalize(input_data)
        # label_feat = np.array(gt_label)
        # #label_feat[label_feat < 40] = -1
        # #print('len: 41 ', len(input_data[label_feat == 41]))
        # tsne_plot(input_data, label_feat, 'svae_onego')
        
        
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))        
            
    
    
    
    
    
    def generating_svae_synthetic(self, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
        num_syn_samples = 300            
        all_attrs = torch.Tensor(dataset.attrs).float().cuda().repeat_interleave(num_syn_samples, dim=0)
        all_labels = torch.tensor([0,1,2,3,4]).long().cuda().repeat_interleave(num_syn_samples)
        print('all_attrs_size: ', all_attrs.size())
        print('all_labels_size: ', all_labels.size())

        self.load_models(epoch)        
        
        with torch.no_grad():
            m, s = self.attr_encoder(all_attrs)        
            z_real = self.reparameterize(m, s).rsample().squeeze()
            synthetic_samples = self.decoder(z_real).detach().cpu().numpy()

        print('synthetic_samples: ', synthetic_samples.shape)

        svae_data = {'train_feat': synthetic_samples,
                     'train_label': all_labels.detach().cpu().numpy()}
        
        print('Saving Data')
        torch.save(svae_data, '/home/sethupathy/openworl_zsl/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/synthetic_data_svae/svae_unseen_data_300.zip')
        
        
        
        
        
        # Visualize:
        # m, s = self.encoder(torch.from_numpy(unseen_feat).cuda())   
        # z_real = self.reparameterize(m, s).rsample().squeeze()
        # z_real = m.squeeze()
        # input_data = z_real.detach().cpu().numpy() #torch.cat((input_data2, input_data3, known_syn_feat), dim=0).detach().cpu().numpy()
        # #input_data = normalize(input_data)
        # label_feat = np.array(unseen_label)
        # label_feat[label_feat < 40] = -1
        # print('len: 41 ', len(input_data[label_feat == 41]))
        # tsne_plot(input_data[label_feat == 41], label_feat[label_feat == 41], 'svae_stage_2')
        
        
        
        # data_dict = {}
        # data_dict['seen_feat'] = np.asarray(seen_feat_split)    
        # data_dict['seen_label_gt'] = np.asarray(seen_label_gt) 
        # data_dict['unseen_feat_split'] = np.asarray(unseen_feat_split) 
        # data_dict['unseen_label_gt'] = np.asarray(unseen_label_gt) 
        # torch.save(data_dict, '/home/sethupathy/openworl_zsl/zsl-openworld/test_classifier_cvae/data_{}.pt'.format(unseen_ood))        
        


class Model_train2(object):
    def __init__(self, 
                 dataset_name,
                 encoder,
                 encoder_prev,
                 decoder,
                 attr_encoder,
                 attr_encoder_prev,
                 attr_decoder,
                 classifier,
                 train_loader,
                 test_loader_unseen,
                 test_loader_seen,
                 criterion,
                 lr = 1e-3,
                 all_attrs = None,
                 epoch = 10000,
                 save_path = "/data/xingyu/wae_lle/experiments/",
                 save_every = 1,
                 iftest = False,
                 ifsample = False,
                 data = None,
                 GZSL = True,
                 zsl_classifier = None
                 ):  
        self.dataset_name = dataset_name
        self.encoder = encoder
        self.encoder_prev = encoder_prev
        self.decoder = decoder
        self.attr_encoder = attr_encoder
        self.attr_encoder_prev = attr_encoder_prev
        self.attr_decoder = attr_decoder
        self.classifier = classifier
        self.zsl_classifier = zsl_classifier
        self.train_loader = train_loader
        self.test_loader_unseen = test_loader_unseen
        self.test_loader_seen = test_loader_seen
           
        self.criterion = criterion
        self.crossEntropy_Loss = nn.NLLLoss()
        
        self.all_attrs = all_attrs
        self.lr = lr
        self.epoch = epoch
        self.save_path = save_path
        self.save_every = save_every
        self.ifsample = ifsample
        self.data = data
        self.GZSL = GZSL
        self.distribution = 'vmf'
        self.sinkhorn = emd.SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        
        if iftest:
            log_dir = '{}/log'.format(self.save_path)
            #general.logger_setup(log_dir, 'results__')
        
        
    def save_checkpoint(self,state, filename = 'checkpoint.pth.tar'):
        torch.save(state, filename)  
         
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
        else:
            raise NotImplemented

        return q_z
        
       
    def compute_acc(self,trues, preds):
        """
        Given true and predicted labels, computes average class-based accuracy.
        """

        # class labels in ground-truth samples
        classes = np.unique(trues)
        # class-based accuracies
        cb_accs = np.zeros(classes.shape, np.float32)
        #ipdb.set_trace()
        for i, label in enumerate(classes):
            inds_ci = np.where(trues == label)[0]

            cb_accs[i] = np.mean(
              np.equal(
              trues[inds_ci],
              preds[inds_ci]
            ).astype(np.float32)
        )
        #ipdb.set_trace()
        return np.mean(cb_accs)   
      
    def training_der(self, data_config, checkpoint = -1, checkpoint_num=None, save_path1=None):
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)
                    
        loss_dict = {'recon_loss': [],
                     'KL_loss': [],
                     'attr_loss': [],
                     'cls_loss': [],
                     'DER_loss': []}
        
        Der_criterion = nn.L1Loss()

        if checkpoint_num != None:
            self.save_path1 = save_path1
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint_num)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint_num)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint_num)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint_num)
            #file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(checkpoint_num)
                
            enc_path = os.path.join(self.save_path1, file_encoder)
            dec_path = os.path.join(self.save_path1, file_decoder)
            attr_enc_path = os.path.join(self.save_path1, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path1, file_attr_decoder)
            #classifier_path = os.path.join(self.save_path, file_classifier)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
            # DER:
            self.encoder_prev.load_state_dict(enc_checkpoint['state_dict'])
            for param in self.encoder_prev.parameters():
                param.requires_grad = False
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            # DER:
            self.attr_encoder_prev.load_state_dict(attr_enc_checkpoint['state_dict'])
            for param in self.attr_encoder_prev.parameters():
                param.requires_grad = False

            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
            
            #classifier_checkpoint = torch.load(classifier_path)
            #self.classifier.load_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.train()
        self.encoder_prev.eval()
        self.decoder.train()
        self.attr_encoder.train() 
        self.attr_encoder_prev.eval()
        self.attr_decoder.train()
        self.classifier.train()
        
        enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
        classifier_optim = optim.Adam(self.classifier.parameters(), lr = self.lr)
              
        enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        classifier_scheduler = StepLR(classifier_optim, step_size=10000, gamma=0.5)
        
        # seen data for der
        seen_data = self.data.seen_data
        seen_data_attr = self.data.seen_data_attr
        seen_data_dataset = TensorDataset(torch.from_numpy(seen_data), torch.from_numpy(seen_data_attr))
        seen_data_loader = DataLoader(seen_data_dataset, batch_size=data_config['der_batch'], shuffle=True)

        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.encoder_prev = self.encoder_prev.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_encoder_prev = self.attr_encoder_prev.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
            self.classifier = self.classifier.cuda()
        print("Begin Training ##############################>>>>>>>>")    
        for epoch in range(checkpoint+1, self.epoch):
            print("epoch: {}".format(epoch))
            # print("train_loader {}".format(len(self.train_loader)))
            # sys.exit()
            
            loss_batch = {'recon_loss': [],
                          'KL_loss': [],
                          'attr_loss': [],
                          'cls_loss': [],
                          'DER_loss': []}
            
            step = 0 
            train_data_iter = iter(self.train_loader)
            for i_batch, sample_batched in enumerate(self.train_loader):                      
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
                # print('unique_label', np.unique(input_label))
                # sys.exit()
              
                batch_size = input_data.size()[0]
                if torch.cuda.is_available():
                    input_data = input_data.float().cuda()
                    input_label = input_label.long().view(-1).cuda()
                    input_attr = input_attr.float().cuda().squeeze()
                    input_der_iter = iter(seen_data_loader)
                    input_der, input_attr_der = next(input_der_iter)
                    input_der, input_attr_der = input_der.float().cuda(), input_attr_der.float().cuda()
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                self.attr_decoder.zero_grad()
                self.classifier.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                # DER:
                m1_current, s1_current = self.encoder(input_der)
                m1_prev, s1_prev = self.encoder_prev(input_der)
                                
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                # DER:
                m2_current, s2_current = self.attr_encoder(input_attr_der)
                m2_prev, s2_prev = self.attr_encoder_prev(input_attr_der)

                # print('der shape:', m1_current.size(), s1_current.size(), m1_prev.size(), s1_prev.size(), m2_current.size(), m2_prev.size())
                # print('der shape:', s2_current.size(), s2_prev.size())
                # sys.exit()

                z_x = z1.rsample()
                z_attr = z2.rsample()
                
                sub_batch_size = 10
                z_x_2 = z1.rsample(sub_batch_size).permute(1,0,2)
                z_attr_2 = z2.rsample(sub_batch_size).permute(1,0,2)
                
            
                z_input = torch.cat((z_attr.squeeze(), z_x),0) 
                label_input = torch.cat((input_label, input_label),0)
             
                cls_out = self.classifier(z_input)
                cls_loss = self.crossEntropy_Loss(cls_out, label_input) 
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                #original code
                x_recon = self.decoder(z_input)
                recon_loss = self.criterion(x_recon, torch.cat((input_data,input_data),0))
                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, torch.cat((input_attr,input_attr),0))
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
     
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)
                #ipdb.set_trace()
            
                KL_loss = dist.mean()

                DER_loss = F.mse_loss(m1_current, m1_prev) + F.mse_loss(s1_current, s1_prev) + F.mse_loss(m2_current, m2_prev) + F.mse_loss(s2_current, s2_prev)
                # DER_loss = Der_criterion(m1_current, m1_prev) + Der_criterion(s1_current, s1_prev) + Der_criterion(m2_current, m2_prev) + Der_criterion(s2_current, s2_prev)
               
                total_loss =  recon_loss *1.0 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0  + DER_loss * data_config['der_weight']
            
                total_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()
                step += 1
            
                with torch.no_grad():
                    loss_batch['attr_loss'].append(attr_loss.data.item())
                    loss_batch['cls_loss'].append(cls_loss.data.item())
                    loss_batch['KL_loss'].append(KL_loss.data.item() * 0.1)
                    loss_batch['recon_loss'].append(recon_loss.data.item())
                    loss_batch['DER_loss'].append(DER_loss.data.item() * data_config['der_weight'])
                
                
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), cls_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))
   
            if epoch % self.save_every ==0: 
            
                write_path = os.path.join(data_config['write_path'], data_config['test_file_name'])
                with open('{}_loss.csv'.format(write_path), mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, 
                                     np.mean(loss_batch['attr_loss']), 
                                     np.mean(loss_batch['cls_loss']), 
                                     np.mean(loss_batch['KL_loss']), 
                                     np.mean(loss_batch['recon_loss']),
                                     np.mean(loss_batch['DER_loss'])])
                
                # loss_dict['attr_loss'].append(np.mean(loss_batch['attr_loss']))
                # loss_dict['cls_loss'].append(np.mean(loss_batch['cls_loss'])) 
                # loss_dict['KL_loss'].append(np.mean(loss_batch['KL_loss'])) 
                # loss_dict['recon_loss'].append(np.mean(loss_batch['recon_loss'])) 
                # loss_dict['DER_loss'].append(np.mean(loss_batch['DER_loss'])) 
                
                
                
                file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict(), 
                     'optimizer': enc_optim.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict(), 
                     'optimizer': dec_optim.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': self.classifier.state_dict(), 
                     'optimizer': classifier_optim.state_dict()}, 
                     file_name_classifier)   
            
    
    def training2(self, checkpoint = -1, checkpoint_num=None, save_path1=None):
        print('*******************************************************************')
        print('Training2')
        print('*******************************************************************')
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)
    
        if checkpoint_num != None:
            print('*******************************************************************')
            print('Previous Weight')
            print('*******************************************************************')
            self.save_path1 = save_path1
            file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(checkpoint_num)
            file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(checkpoint_num)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(checkpoint_num)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(checkpoint_num)
            #file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(checkpoint_num)
                
            enc_path = os.path.join(self.save_path1, file_encoder)
            dec_path = os.path.join(self.save_path1, file_decoder)
            attr_enc_path = os.path.join(self.save_path1, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path1, file_attr_decoder)
            #classifier_path = os.path.join(self.save_path, file_classifier)
                
            enc_checkpoint = torch.load(enc_path)
            self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = torch.load(dec_path)
            self.decoder.load_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = torch.load(attr_enc_path)
            self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = torch.load(attr_dec_path)
            self.attr_decoder.load_state_dict(attr_dec_checkpoint['state_dict'])
            
            #classifier_checkpoint = torch.load(classifier_path)
            #self.classifier.load_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.train()
        self.decoder.train()
        self.attr_encoder.train() 
        self.attr_decoder.train()
        self.classifier.train()
        
        enc_optim = optim.Adam(self.encoder.parameters(), lr = self.lr)
        dec_optim = optim.Adam(self.decoder.parameters(), lr = self.lr)
        attr_enc_optim = optim.Adam(self.attr_encoder.parameters(), lr = self.lr)
        attr_dec_optim = optim.Adam(self.attr_decoder.parameters(), lr = self.lr)
        classifier_optim = optim.Adam(self.classifier.parameters(), lr = self.lr)
              
        enc_scheduler = StepLR(enc_optim, step_size=10000, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=10000, gamma=0.5)
        attr_enc_scheduler = StepLR(attr_enc_optim, step_size=10000, gamma=0.5)
        attr_dec_scheduler = StepLR(attr_dec_optim, step_size=10000, gamma=0.5)
        classifier_scheduler = StepLR(classifier_optim, step_size=10000, gamma=0.5)
        
        
        # Take seen data as OOD:
        seen_ood_feat = np.vstack([self.data.train_set_, self.data.val_set])
        seen_ood_labels = np.asarray([5 for i in range(len(seen_ood_feat))]) # np.vstack([self.data.train_labels_, self.data.val_labels])
        
        seen_ood_dataset = TensorDataset(torch.from_numpy(seen_ood_feat), torch.from_numpy(seen_ood_labels))
        seen_ood_dataloader = DataLoader(seen_ood_dataset, shuffle=True, batch_size=128)
        

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attr_encoder = self.attr_encoder.cuda()
            self.attr_decoder = self.attr_decoder.cuda()
            self.classifier = self.classifier.cuda()
        print("Begin Training ##############################>>>>>>>>")    
        for epoch in range(checkpoint+1, self.epoch):
            print("epoch: {}".format(epoch))
            # print("train_loader {}".format(len(self.train_loader)))
            # sys.exit()
            step = 0 
            train_data_iter = iter(self.train_loader)
            for i_batch, sample_batched in enumerate(self.train_loader):                      
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
                # print('unique_label', np.unique(input_label))
                # sys.exit()

                ood_iter = iter(seen_ood_dataloader)
                ood_feat, ood_label = next(ood_iter)
              
                batch_size = input_data.size()[0]
                if torch.cuda.is_available():
                    input_data = input_data.float().cuda()
                    input_label = input_label.long().view(-1).cuda()
                    input_attr = input_attr.float().cuda().squeeze()
                    ood_feat, ood_label = ood_feat.float().cuda(), ood_label.long().cuda()
                        
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.attr_encoder.zero_grad()
                self.attr_decoder.zero_grad()
                self.classifier.zero_grad()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)
                m2, s2 = self.attr_encoder(input_attr)
                z2 = self.reparameterize(m2, s2)
                
                # Seen as OOD
                m1_ood, s1_ood = self.encoder(ood_feat)
                z1_ood = self.reparameterize(m1_ood, s1_ood)
                z_ood = z1_ood.rsample()

                z_x = z1.rsample()
                z_attr = z2.rsample()
                
                sub_batch_size = 10
                z_x_2 = z1.rsample(sub_batch_size).permute(1,0,2)
                z_attr_2 = z2.rsample(sub_batch_size).permute(1,0,2)
                
            
                z_input = torch.cat((z_attr.squeeze(), z_x),0) 
                label_input = torch.cat((input_label, input_label),0)
             
                cls_out = self.classifier(z_input)
                cls_out_ood = self.classifier(z_ood)
                cls_loss = self.crossEntropy_Loss(cls_out, label_input) + 0.5 * self.crossEntropy_Loss(cls_out_ood, ood_label)
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                #original code
                x_recon = self.decoder(z_input)
                x_recon_ood = self.decoder(z_ood)
                recon_loss = self.criterion(x_recon, torch.cat((input_data,input_data),0)) + self.criterion(x_recon_ood, ood_feat)
                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, torch.cat((input_attr,input_attr),0))
                
                if torch.cuda.is_available():
                    z_attr = z_attr.cuda()
     
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)
                #ipdb.set_trace()
            
                KL_loss = dist.mean()
               
                total_loss =  recon_loss *1.0 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0  
            
                total_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()
                step += 1
            
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.data.item(), KL_loss.data.item(), attr_loss.data.item(), cls_loss.data.item(), s1.mean().data.item(), s2.mean().data.item(), torch.dot(z_x[1,:], z_attr.squeeze()[1,:]).data.item()))
   
            if epoch % self.save_every ==0: 
            
                file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pth.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict(), 
                     'optimizer': enc_optim.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict(), 
                     'optimizer': dec_optim.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': self.classifier.state_dict(), 
                     'optimizer': classifier_optim.state_dict()}, 
                     file_name_classifier) 
    
    def search_thres_by_sample(self, attrs, n = 10000):
        min_thres = 100
        m, s = self.attr_encoder(attrs)
      
        z = []
        for i in range(n):
            z_fake = self.reparameterize(m, s).rsample()
            dist = F.cosine_similarity(m, z_fake)
            z.append(z_fake)
            thres = dist.min()
            if min_thres > thres:
                min_thres = thres
        
        return min_thres
        
    def load_models(self, epoch):
        file_encoder = 'Checkpoint_{}_Enc.pth.tar'.format(epoch)
        file_decoder = 'Checkpoint_{}_Dec.pth.tar'.format(epoch)
        file_attr_encoder = 'Checkpoint_{}_attr_Enc.pth.tar'.format(epoch)  
        file_classifier = 'Checkpoint_{}_classifier.pth.tar'.format(epoch)  
        enc_path = os.path.join(self.save_path, file_encoder)
        dec_path = os.path.join(self.save_path, file_decoder)
        attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
        classifier_path = os.path.join(self.save_path, file_classifier)
        enc_checkpoint = torch.load(enc_path)
        self.encoder.load_state_dict(enc_checkpoint['state_dict'])
        dec_checkpoint = torch.load(dec_path)
        self.decoder.load_state_dict(dec_checkpoint['state_dict'])
        attr_enc_checkpoint = torch.load(attr_enc_path)
        self.attr_encoder.load_state_dict(attr_enc_checkpoint['state_dict'])
        classifier_checkpoint = torch.load(classifier_path)
        self.classifier.load_state_dict(classifier_checkpoint['state_dict'])       
        
        # Load the ZSL classifiers. These ZSL classifiers can be replaced by any SOTA models! 
        if self.dataset_name == 'AWA1':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/awa1/Checkpoint_24_Classifier.pth.tar")
        elif self.dataset_name == 'AWA2':
            zsl_classifier_checkpoint = torch.load("/home/sethupathy/openworl_zsl/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/zsl_models/awa2_Checkpoint_9_Classifier.pth.tar")
        elif self.dataset_name == 'CUB':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/cub/Checkpoint_7_Classifier.pth.tar")
        elif self.dataset_name == 'FLO':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/flo/Checkpoint_24_Classifier.pth.tar")
        elif self.dataset_name == 'SUN':
            zsl_classifier_checkpoint = torch.load("/home/svc6/origin/cvpr18xian/checkpoint/sun/Checkpoint_14_Classifier.pth.tar")
        
        self.zsl_classifier.load_state_dict(zsl_classifier_checkpoint['state_dict'])
        
        self.encoder.eval()
        self.decoder.eval()
        self.attr_encoder.eval()  
        self.zsl_classifier.eval()      
        if torch.cuda.is_available():
             self.encoder, self.decoder, self.attr_encoder, self.zsl_classifier, self.classifier = self.encoder.cuda(), self.decoder.cuda(), self.attr_encoder.cuda(), self.zsl_classifier.cuda(), self.classifier.cuda()
        
    
     
        
        
    def search_thres_by_traindata(self, epoch, dataset = None, n = 0.95):
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = dataset.seen_labels
        unseen_labels = dataset.unseen_labels
        self.load_models(epoch)

        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []    
        all_anchors = self.attr_encoder(all_attrs)[0]      
        seen_idx = seen_labels - 1
        unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors #all_anchors[seen_idx.tolist(),:]
        #unseen_anchors = all_anchors[unseen_idx.tolist(),:]       
        seen_count = 0
        seen_all = 0
        unseen_count = 0
        unseen_all = 0
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        
        for i_batch, sample_batched in enumerate(self.train_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']
            batch_size = input_data.size()[0]
            if torch.cuda.is_available():
                input_data = input_data.float().cuda()
                input_label = input_label.cuda()  
                input_attr = input_attr.float().cuda()  
                                
            m, s = self.encoder(input_data)   
            #z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                kk = input_label[k]+1 #input_label[k,:]+1
                z_tile = z_real[k,:].repeat(seen_anchors.shape[0]).view(seen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, seen_anchors)
                if min_thres>dist.max():
                    min_thres = dist.max()
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
            
        dist_array = np.array(dist_list)
        idx = dist_array.shape[0] * (1.0 - n)
        thres  = np.sort(dist_array)[int(idx)]

      
        return thres 
   
    
    
   