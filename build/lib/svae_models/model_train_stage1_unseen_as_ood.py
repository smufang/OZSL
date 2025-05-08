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
from sklearn.metrics import top_k_accuracy_score, f1_score, confusion_matrix

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
    plt.legend(labels=['Unseen', 'Unknown'], loc='lower left')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    # plt.axis('off')
    # plt.get_xaxis().set_visible(False)
    # plt.get_yaxis().set_visible(False)
    # plt.tight_layout()
    fig = scatterplot.get_figure()
    # axes = scatterplot.get_axes()
    # scatterplot.get_xaxis().set_visible(False)
    # scatterplot.get_yaxis().set_visible(False)
    fig.savefig("{}.png".format(file_name)) 
    plt.close()

    # ax.patch.set_edgecolor('black')  
    # ax.patch.set_linewidth('1')
    
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

def confusion_mat_rate(pred, target_label):
    print("unique in pred for confusion matrix:", np.unique(pred))

    conf_matrix = confusion_matrix(target_label, pred)
    # conf_matrix_percent = np.asarray(conf_matrix) * 1.0
    conf_matrix = np.array(conf_matrix) * 1.0
    print(conf_matrix)
    # conf_matrix = conf_matrix[1:,1:]
    print('conf_matrix', conf_matrix.shape)
    
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)

    print('FP', FP)
    print('FN', FN)
    print('TP', TP)
    print('TN', TN)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print('false_positive_rate:', FPR)
    print('Sensitivity:', TPR)
    print('Percision:', PPV)



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
            save_path = '{}_{}_{}'.format('../../gzsl_svae/experiments/AWA2_synthetic_cvae300',512, 64)
            classifier_path = os.path.join(save_path, file_classifier)
            classifier_checkpoint = torch.load(classifier_path)
            print("i am here")
        else:
            classifier_checkpoint = torch.load(classifier_path)
        self.classifier.load_state_dict(classifier_checkpoint['state_dict'])       
        
        # Load the ZSL classifiers. These ZSL classifiers can be replaced by any SOTA models! 
        if self.dataset_name == 'AWA1':
            zsl_classifier_checkpoint = torch.load("../../A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/zsl_models/awa1_Checkpoint_24_Classifier.pth.tar")
        elif self.dataset_name == 'CUB':
            zsl_classifier_checkpoint = torch.load("../../A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/zsl_models/cub_Checkpoint_7_Classifier.pth.tar")
        elif self.dataset_name == 'FLO':
            zsl_classifier_checkpoint = torch.load("../../A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/zsl_models/flo_Checkpoint_24_Classifier.pth.tar")
        
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
        
    
    def testing_split_ood_synthetic_seed(self, dataset_config, epoch, seen_classes, known_unseen_classes, dataset = None, threshold = 0.99):
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
        
        
        
        
        
        checkpoint1 = torch.load('./data_split_seed/{}_data_seen_seed{}.pt'.format(dataset_config["dataset_name"], self.seed))
        seen_feat1 = checkpoint1['seen_feat']
        seen_label1 = checkpoint1['seen_label_gt']
        unseen_feat1 = checkpoint1['unseen_feat_split']
        unseen_label1 = checkpoint1['unseen_label_gt']

        checkpoint2 = torch.load('./data_split_seed/{}_data_unseen_seed{}.pt'.format(dataset_config["dataset_name"], self.seed))
        seen_feat2 = checkpoint2['seen_feat']
        seen_label2 = checkpoint2['seen_label_gt']
        unseen_feat2 = checkpoint2['unseen_feat_split']
        unseen_label2 = checkpoint2['unseen_label_gt']

        #if dataset_config["dataset_name"] == 'FLO':
        if len(seen_feat2) == 0:
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
        try:
            total_ood = len(gt_label[gt_label == all_class])
            ood_split = len(ood_label_gt[ood_label_gt == all_class])
            print("ood_percentage: ", ood_split/total_ood)
            result['ood_acc'] = ood_split/total_ood
            result['ood_top1'] = ood_split/total_ood
            
            result['ood_F1'] = None ##########!!!!!!!!!!!!!
            print()
        except:
            print("ood_acc:", None)
            result['ood_acc'] = None
            print('ood_top1: ', None)
            result['ood_top1'] = None
            print('F1_score_ood: ', None)
            result['ood_F1'] = None

        
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
          
            result['unseen_F1'] = None ##########!!!!!!!!!!!!!
            print()
        except:
            print("unseen_acc:", None)
            result['unseen_acc'] = None
            print('acc_unseen_top1: ', None)
            result['unseen_top1'] = None
            print('F1_score_unseen: ', None)
            result['unseen_F1'] = None

        return result
        
        
              
     
       
    

   

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
                 zsl_classifier = None,
                 seed = None
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

        self.seed = seed
        
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
                with open('{}_loss_seed{}.csv'.format(write_path, self.seed), mode='a') as file:
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
            zsl_classifier_checkpoint = torch.load("../zsl/awa1/Checkpoint_24_Classifier.pth.tar")
        elif self.dataset_name == 'CUB':
            zsl_classifier_checkpoint = torch.load("../zsl/cub/Checkpoint_7_Classifier.pth.tar")
        elif self.dataset_name == 'FLO':
            zsl_classifier_checkpoint = torch.load("../zsl/flo/Checkpoint_24_Classifier.pth.tar")
        
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
   
    
    
   