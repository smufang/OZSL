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
                 zsl_classifier = None,
                 seed = None
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

        self.seed = seed
        # assert self.seed != None
        
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
                # total_loss =  recon_loss *0.1 + KL_loss * 0.1  + attr_loss *1.0 + cls_loss* 1.0  
            
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
            zsl_classifier_checkpoint = torch.load("../zsl_models/awa1_Checkpoint_24_Classifier.pth.tar")
        elif self.dataset_name == 'CUB':
            zsl_classifier_checkpoint = torch.load("../zsl_models/cub_Checkpoint_7_Classifier.pth.tar")
        elif self.dataset_name == 'FLO':
            zsl_classifier_checkpoint = torch.load("../zsl_models/flo_Checkpoint_24_Classifier.pth.tar")
        
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
        torch.save(data_dict, './data_split_seed/{}_data_{}_seed{}.pt'.format(self.dataset_name, test_class, self.seed))
            
         
        
    
