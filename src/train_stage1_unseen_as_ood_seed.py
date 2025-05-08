import numpy as np
import torch
import os
import torch.nn as nn
import ipdb
import yaml
import argparse
from shutil import copyfile
from utilis_svae import datasets_stage1_unseen_as_ood as datasets
from svae_models import model_train_stage1_unseen_as_ood as model_train
from svae_models import models

from torch.utils.data import Dataset, DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('template_path', type=str)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--seed2', type=int, required=True)
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print('seed:', args.seed)

    with open(args.template_path, 'r') as f:
        template = yaml.safe_load(f)
    dataset_config = template['dataset'] 
    model_config = template['model']
    train_config = template['train']
    save_path = '{}_{}_{}'.format(dataset_config['save_path']+'_seed'+str(args.seed),model_config['mid_size'], model_config['hidden_size'])

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    config_dir = os.path.join(save_path, 'config')
    log_dir = os.path.join(save_path, 'log')
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
        
    config_copy = '{}/{}'.format(config_dir, args.template_path)
    copyfile(args.template_path, config_copy)
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    '''    
    basis_dir = '{}/basis'.format(train_config['res_dir'])
    if not os.path.exists(basis_dir):
        os.mkdir(basis_dir)
    '''
   
    if dataset_config['dataset_name'] == 'ImageNet':
        data = datasets.Imagenet(dataset_config['dataset_name'],
                            dataset_config['data_path'])  
    else:
     
        data = datasets.AwA2(dataset_config['dataset_name'],
                            dataset_config['data_path'],
                            dataset_config,
                            args.seed2)                      
    
    
    dataset_setup_train = datasets.Dataset_setup(
                            data = data.train_set,
                            attrs = data.attrs ,
                            labels = data.train_labels 
                            )
        
    # dataset_setup_val = datasets.Dataset_setup(
    #                         data = data.val_set,
    #                         attrs = data.attrs ,
    #                         labels = data.val_labels 
    #                         )
    # dataset_setup_test_seen = datasets.Dataset_setup(
    #                         data = data.test_seen_set,
    #                         attrs = data.attrs ,
    #                         labels = data.test_seen_labels 
    #                         )
       
    # dataset_setup_test_unseen = datasets.Dataset_setup(
    #                         data = data.test_unseen_set,
    #                         attrs = data.attrs ,
    #                         labels = data.test_unseen_labels 
    #                         )

  
    
    
    
    dataset_loader_train = torch.utils.data.DataLoader(dataset_setup_train, batch_size = train_config['batch_size'], shuffle= True) #, num_workers = 4)
    # dataset_loader_val = torch.utils.data.DataLoader(dataset_setup_val, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    # dataset_loader_test_seen = torch.utils.data.DataLoader(dataset_setup_test_seen, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    # dataset_loader_test_unseen = torch.utils.data.DataLoader(dataset_setup_test_unseen, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    
    
    
    # Models
        
    attr_encoder = models.Attr_Encoder(model_config['attr_size'], model_config['mid_size'], model_config['hidden_size'])
    attr_decoder = models.Attr_Decoder(model_config['hidden_size'], model_config['mid_size'], model_config['attr_size'])
    encoder = models.Encoder(model_config['input_size'], model_config['mid_size'], model_config['hidden_size'])
    decoder = models.Decoder(model_config['hidden_size'], model_config['mid_size'], model_config['input_size'])

        
    # Classifier
    
    if dataset_config['GZSL']:
        classifier = models.LINEAR_LOGSOFTMAX(model_config['hidden_size'], model_config['classes'])
        # cl2 = models.LINEAR_LOGSOFTMAX(model_config['hidden_size'], 2)
    else:
        classifier = models.LINEAR_LOGSOFTMAX(model_config['hidden_size'], data.unseen_labels.shape[0])
        
    print(attr_encoder)
    print(attr_decoder)
    print(encoder)
    print(decoder)
    print(classifier)

    if not dataset_config['use_der']:
        model_train_obj = model_train.Model_train(
                                    dataset_config['dataset_name'],
                                    encoder,
                                    decoder,
                                    attr_encoder,
                                    attr_decoder,
                                    classifier,
                                    dataset_loader_train,
                                    dataset_loader_train,
                                    dataset_loader_train,
                                    criterion = nn.L1Loss(),
                                    lr = 1e-4,
                                    all_attrs = data.attrs,
                                    epoch = train_config['epoch'],
                                    save_path = save_path,
                                    save_every = train_config['save_every'],
                                    ifsample = model_config['ifsample'],
                                    data = data,
                                    GZSL = dataset_config['GZSL']
                                    )
                                  
    
    import time
    t0 = time.time()
    

    if dataset_config['dataset_name'] == 'FLO':
        check_num = 320    

    
    if dataset_config['use_der']:
        attr_encoder_prev = models.Attr_Encoder(model_config['attr_size'], model_config['mid_size'], model_config['hidden_size'])
        encoder_prev = models.Encoder(model_config['input_size'], model_config['mid_size'], model_config['hidden_size'])
        
        model_train_obj = model_train.Model_train2(
                                  dataset_config['dataset_name'],
                                  encoder,
                                  encoder_prev,
                                  decoder,
                                  attr_encoder,
                                  attr_encoder_prev,
                                  attr_decoder,
                                  classifier,
                                  dataset_loader_train,
                                  dataset_loader_train,
                                  dataset_loader_train,
                                  criterion = nn.L1Loss(),
                                  lr = 1e-4,
                                  all_attrs = data.attrs,
                                  epoch = train_config['epoch'],
                                  save_path = save_path,
                                  save_every = train_config['save_every'],
                                  ifsample = model_config['ifsample'],
                                  data = data,
                                  GZSL = dataset_config['GZSL'],
                                  seed = args.seed
                                  )
                                  
        save_path1 = '{}_{}_{}'.format('../../gzsl_svae_test/experiments/{}_seed{}'.format(dataset_config['dataset_name'], args.seed),model_config['mid_size'], model_config['hidden_size'])
        model_train_obj.training_der(dataset_config, train_config['check_point'], checkpoint_num=check_num, save_path1=save_path1)
    

    t1 = time.time()
    total = t1-t0
    print()
    print("Time Taken is...:", total)
   