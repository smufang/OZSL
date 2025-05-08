import numpy as np
import torch
import scipy.io
import os
import ipdb
import pickle
import h5py
#from utils import LLE_utils
#from utils import KNN_utils
from torch.utils.data import Dataset, DataLoader
import scipy.io
from scipy import io
import sys

class Dataset_setup(Dataset):
    def __init__(self,data, attrs, labels):
        self.data = data
        self.attrs = attrs
        self.labels = labels
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        sample_idx = self.data[idx,:]
        attr_idx = self.labels[idx].astype('int16') -1
        attr = self.attrs[attr_idx,:]
        sample = {'feature': sample_idx, 'attr': attr, 'label': attr_idx}
        
        return sample 

class Dataset_setup2(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        sample_idx = self.data[idx,:]
        labels_idx = self.labels[idx]
        sample = {'feature': sample_idx, 'label': labels_idx}
        return sample
        
class Dataset_setup_batch(Dataset):
    def __init__(self, data, attrs, labels):
        self.data = data
        self.attrs = attrs
        self.labels = labels
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_idx = self.data[idx]
        attr_idx = self.labels[idx].astype('int16') -1
        attr_ = self.attrs[attr_idx[0]]
        attr = np.tile(attr_, (sample_idx.shape[0],1))
        sample = {'feature': sample_idx, 'attr': attr, 'label': attr_idx}
        
        return sample 
        

class Imagenet(object):
    def __init__(self,
                dataset_name,
                data_path,
                ifnorm = True):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.ifnorm = ifnorm
        self.prepare_data()
        
    def norm_data(self):
        for i in range(self.attrs.shape[0]):
            print('{} {}'.format(i,np.linalg.norm(self.attrs[i,:])))
            self.attrs[i,:] = self.attrs[i,:]/np.linalg.norm(self.attrs[i,:])
        print('norm attributes done!')
    
        for i in range(self.features.shape[0]):
            self.features[i,:] = self.features[i,:]/np.linalg.norm(self.features[i,:])
        
        for i in range(self.features_val.shape[0]):
            self.features_val[i,:] = self.features_val[i,:]/np.linalg.norm(self.features_val[i,:])    
         
        print('norm features done!')
        
        
    
    def prepare_data(self):
        feature_path = os.path.join(self.data_path, "ILSVRC2012_res101_feature.mat")
        attr_path = os.path.join(self.data_path, "ImageNet_w2v.mat")        
        with h5py.File(attr_path, 'r') as f:
            attr_keys = list(f.keys())
            '''
            no_w2v_loc = f['no_w2v_loc']
            wnids = f['wnids']
            words = f['words']
            '''
            w2v = f['w2v']
            self.attrs = w2v[:].T
            

        with h5py.File(feature_path, 'r') as f: 
            dataset_keys = list(f.keys())
            self.features = f['features'][:].T
            self.features_val = f['features_val'][:]
            self.labels = f['labels'][:].T
            self.labels_val = f['labels_val'][:].T     
            #self.visual_features = features 
            #self.visual_labels = labels 
                  
        '''            
        if self.ifnorm:
            self.norm_data()
        '''
        
        train_idx = np.where(self.labels <= 200)[0]
        test_seen_idx = np.where(self.labels_val <=200)[0]
        test_unseen_idx = np.where(self.labels_val>900)[0]
        
        self.train_set = self.features[train_idx, :]
        self.train_labels = self.labels[train_idx, :]
        
        self.test_seen_set = self.features_val[test_seen_idx, :]
        self.test_seen_labels = self.labels_val[test_seen_idx, :]
        
        self.test_unseen_set = self.features_val[test_unseen_idx, :]
        self.test_unseen_labels = self.labels_val[test_unseen_idx, :]
        
        self.val_set = self.test_seen_set
        self.val_labels = self.test_seen_labels
        
        self.seen_labels = np.array(list(range(1,200)))
        self.unseen_labels = np.array(list(range(901, 1000)))
        


        
class AwA2(object):
    def __init__(self,
                 dataset_name,
                 data_path,
                 data_config,
                 seed2,
                 ifnorm = True):
                 
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.data_config = data_config
        self.seed2 = seed2
        self.ifnorm = ifnorm
        self.prepare_data()
    
    def norm_data(self):
        for i in range(self.visual_features.shape[0]):
            self.visual_features[i,:] = self.visual_features[i,:]/np.linalg.norm(self.visual_features[i,:]) * 1.0
        print('norm features done!')
        
    def prepare_data(self):
        
        feature_path = os.path.join(self.data_path, "res101.mat")
        attr_path = os.path.join(self.data_path, "att_splits.mat")
        
        features = scipy.io.loadmat(feature_path)
        attr = scipy.io.loadmat(attr_path)
        self.visual_features = features['features'].T
        self.visual_labels = features['labels']
        
        # Load Attributes and Data:
        if self.data_config['use_orginal_attr']:
            sys.exit()  ##################!!!!!!!!!!!!!!!!!!!!!!!
            print('Using orginal attributes:')
            self.attrs = attr['att'].T * 1.0

            # Load Data
            self.train_loc = attr['train_loc']
            self.val_loc = attr['val_loc'] 
            self.trainval_loc = attr['trainval_loc']
            self.test_seen_loc = attr['test_seen_loc']
            self.test_unseen_loc = attr['test_unseen_loc']
            
            
            if self.ifnorm:
                self.norm_data()
            
            self.train_set_ = self.visual_features[self.train_loc.reshape(-1)-1,:]
            self.train_labels_ = self.visual_labels[self.train_loc.reshape(-1)-1,:]
            
            self.val_set = self.visual_features[self.val_loc.reshape(-1)-1,:]
            self.val_labels = self.visual_labels[self.val_loc.reshape(-1)-1,:]
            
            self.trainval_set =  self.visual_features[self.trainval_loc.reshape(-1)-1,:]
            self.trainval_labels = self.visual_labels[self.trainval_loc.reshape(-1)-1,:]                
                
            self.test_seen_set = self.visual_features[self.test_seen_loc.reshape(-1)-1,:]
            self.test_seen_labels = self.visual_labels[self.test_seen_loc.reshape(-1)-1,:]  
            self.test_unseen_set = self.visual_features[self.test_unseen_loc.reshape(-1)-1,:]
            self.test_unseen_labels = self.visual_labels[self.test_unseen_loc.reshape(-1)-1,:]       
              
            self.train_set = np.vstack([self.train_set_, self.val_set])
            self.train_labels = np.vstack([self.train_labels_, self.val_labels]) 

            self.seen_labels = np.unique(self.test_seen_labels).astype('int16')
            self.unseen_labels = np.unique(self.test_unseen_labels).astype('int16')

            if self.data_config['domain_loss']:
                print('Using Domain Loss:')
                train_data = torch.load(os.path.join(self.data_config['syn_data_root'], self.data_config['domain_loss_data']))
                self.Unseen_Data = train_data['train_feat']
                self.Unseen_Labels = train_data['train_label']
                
                attr_data = torch.load(self.data_config['attr_path'])
                self.domain_attrs = attr_data['unseen_attr']

        
        else:
            print('Using saved attributes:')
            attr_data = torch.load(self.data_config['attr_path']+'{}_attr_seed{}.pth'.format(self.dataset_name, self.seed2))
            if self.data_config['attrs_type'] == 'seen':
                self.attrs = attr_data['seen_attr']
                assert(len(self.attrs) == self.data_config['seen_classes'])
                print('Using seen attr:')
                sys.exit() ###########!!!!!!!!!!!!!!!!
            
            elif self.data_config['attrs_type'] == 'unseen':
                self.attrs = attr_data['unseen_attr']
                assert(len(self.attrs) == self.data_config['unseen_classes'])
                print('Using unseen attr:')
                sys.exit() #########!!!!!!!!!!!!!!!
            
            elif self.data_config['attrs_type'] == 'both':
                if self.dataset_name == 'CUB' or self.dataset_name == 'FLO':
                    self.attrs = np.concatenate((attr_data['seen_attr'].numpy(), attr_data['unseen_attr'].numpy()), axis=0) # used .numpy()
                elif self.dataset_name == 'AWA1' or self.dataset_name == 'AWA2':
                    self.attrs = np.concatenate((attr_data['seen_attr'], attr_data['unseen_attr']), axis=0)
                
                assert(len(self.attrs) == self.data_config['seen_classes'] + self.data_config['unseen_classes'])
                print('Using seen and unseen attr:')
                
            else:
                print("Wrong attr info")
                sys.exit()

            # Load Data
            #train_data = torch.load(os.path.join(self.data_config['syn_data_root'], self.data_config['syn_data']))
            train_data = torch.load(os.path.join(self.data_config['syn_data_root'], '{}_unseen_data_seed{}.pth'.format(self.data_config['dataset_name'], self.seed2)))
            self.train_set = train_data['train_feat']
            self.train_labels =  train_data['train_label'] + 1 + self.data_config['seen_classes']
            len_train_lables = len(self.train_labels)
            

            seen_data_path = os.path.join(self.data_config['syn_data_root'], "{}_seen_data_seed{}.pth".format(self.data_config['dataset_name'], self.seed2))
            train_data_seen = torch.load(seen_data_path)
            seen_data = train_data_seen['train_feat']
            seen_data_label =  train_data_seen['train_label'] + 1
            len_seen_data_label = len(seen_data_label)
            

            self.train_set = np.concatenate((seen_data, self.train_set), axis=0)
            self.train_labels = np.concatenate((seen_data_label, self.train_labels), axis=0)
            print('self.train_labels: ', self.train_labels.shape)
            assert len(self.train_labels) == len_train_lables + len_seen_data_label


                        
            if self.data_config['use_percent']:
                train_idx = np.random.choice(len(self.train_set), int(len(self.train_set) * self.data_config['percent']))
                self.train_set = self.train_set[train_idx]
                self.train_labels = self.train_labels[train_idx]
                print('percent useage', len(self.train_set))
                sys.exit()  #########!!!!!!!!!!!!!!!           
                        
            
            assert(len(np.unique(self.train_labels)) == self.data_config['num_attr'])
            print('train_label', np.unique(self.train_labels))
            print('data_set_size: ', len(self.train_labels))
            #sys.exit()

            #DER:
            if self.data_config['use_der']:
                print('Using DER:')
                seen_data_path = os.path.join(self.data_config['syn_data_root'], "{}_seen_data_seed{}.pth".format(self.data_config['dataset_name'], self.seed2))
                # train_data_seen = torch.load('/home/sethupathy/openworl_zsl/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/synthetic_data_svae/seen_data.pth')
                train_data_seen = torch.load(seen_data_path)
                self.seen_data = train_data_seen['train_feat']
                seen_data_label =  train_data_seen['train_label']
                
                if self.dataset_name == 'CUB' or self.dataset_name == 'FLO':
                    self.seen_data_attr = attr_data['seen_attr'][seen_data_label].numpy() # used .numpy()
                elif self.dataset_name == 'AWA1' or self.dataset_name == 'AWA2':
                    self.seen_data_attr = attr_data['seen_attr'][seen_data_label] # used .numpy()
                
                assert len(np.unique(seen_data_label)) == self.data_config['seen_classes'] and np.max(seen_data_label) == self.data_config['seen_classes'] - 1
                print('seen_data_label in dataset: ', np.unique(seen_data_label))
                sys.exit()
            
            # if self.data_config['domain_loss']:
            #     print('Using Domain Loss:')
                
            #     ###########################################
            #     DATASET = 'AWA2'
            #     print(f'<=============== Loading data for {DATASET} ===============>')
            #     DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
            #     DATA_DIR = f'/home/sethupathy/openworl_zsl/Datasets/{DATASET}'
            #     data = io.loadmat(f'{DATA_DIR}/res101.mat')
            #     attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
            #     feats = data['features'].T.astype(np.float32)
            #     labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
            #     train_idx = attrs_mat['trainval_loc'].squeeze() - 1
                
            #     seen_classes = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 31, 32, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47, 48]
            #     seen_train_feat = feats[train_idx]
            #     seen_train_label = np.asarray([seen_classes.index(labels[i]) for i in train_idx])

            #     ##############################################
                               
            #     self.Unseen_Data = np.array(seen_train_feat)
            #     self.Unseen_Labels = np.array(seen_train_label)
                
            #     attr_data = torch.load(self.data_config['attr_path'])
            #     self.domain_attrs = attr_data['seen_attr']

            
            
            
            # self.seen_labels = np.unique(self.test_seen_labels).astype('int16')
            # self.unseen_labels = np.unique(self.test_unseen_labels).astype('int16')     
            
            # train_data = torch.load('/home/sethupathy/openworl_zsl/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning-master/synthetic_data_svae/unseen_data.pth')
            # self.Unseen_Data = train_data['train_feat']
            # self.Unseen_Labels = train_data['train_label'] + 40
            
            #self.test_seen_set = self.trainval_set
            #self.test_seen_labels = self.trainval_labels 
        

        
        '''
        # binary labels for visualization     
      
        self.test_seen_labels2 = np.ones(self.test_seen_labels.shape[0]).reshape(-1,1).astype('int16')
        self.test_unseen_labels2 = np.ones(self.test_unseen_labels.shape[0]).reshape(-1,1)*2
        self.test_unseen_labels2 = self.test_unseen_labels2.astype('int16')
        
        self.test_unseen_set = np.vstack([self.test_unseen_set, self.test_seen_set])
        self.test_unseen_labels = np.vstack([self.test_unseen_labels2, self.test_seen_labels2])
        '''
 

        
    
        
        
    
    
        