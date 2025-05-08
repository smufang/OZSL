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
import csv

np.random.seed(1)
torch.manual_seed(1)


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
    basis_config = template['basis']
    basis_dir = '{}/basis'.format(save_path)
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
                            seed2 = args.seed2)
    
       
                 
    dataset_setup_train = datasets.Dataset_setup(
                            data = data.train_set,
                            attrs = data.attrs ,
                            labels = data.train_labels 
                            )
        
    

    dataset_loader_train = torch.utils.data.DataLoader(dataset_setup_train, batch_size = train_config['batch_size'], shuffle= True, num_workers = 8)
    
    
    
    # Models
        
    attr_encoder = models.Attr_Encoder(model_config['attr_size'], model_config['mid_size'], model_config['hidden_size'])
    attr_decoder = models.Attr_Decoder(model_config['hidden_size'], model_config['mid_size'], model_config['attr_size'])
   
    encoder = models.Encoder(model_config['input_size'], model_config['mid_size'], model_config['hidden_size'])
    decoder = models.Decoder(model_config['hidden_size'], model_config['mid_size'], model_config['input_size'])

        
    # Classifier
    
    if dataset_config['GZSL']:
        classifier = models.LINEAR_LOGSOFTMAX(model_config['hidden_size'], model_config['classes'])
        
    else:
        classifier = models.LINEAR_LOGSOFTMAX(model_config['hidden_size'], data.unseen_labels.shape[0])
        
    
    if dataset_config['dataset_name'] == 'FLO':
        zsl_classifier = models.LINEAR_LOGSOFTMAX(2048, 20)
    
    
    print(attr_encoder)
    print(attr_decoder)
    print(encoder)
    print(decoder)
    print(classifier)

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
                                GZSL = dataset_config['GZSL'],
                                zsl_classifier = zsl_classifier,
                                seed = args.seed
                                )
   
    
    #####################################################################################################################
    # Test Single epoch:
    for i in range(10,100,10):  #(10,1000,100)
    
    
        # FLO
        if dataset_config['dataset_name'] == 'FLO':
            seen_classes = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
            class_split = torch.load('../Stage2_data/FLO_class_split_seed{}.pth'.format(args.seed2))
            known_unseen_classes = class_split['known_unseen_class']
            assert len(seen_classes + known_unseen_classes) == 92

        
        test_epoch = i
        unseen_top1 = []
        ood_top1 = []
        seen_ood = []
        tpr = 0.65
        threshold = tpr 
        print("the threshold is:", threshold)

        # model_train_obj.generating_svae_synthetic(test_epoch, seen_classes, known_unseen_classes, dataset = data, threshold = 0.80)
        # result = model_train_obj.testing_split_ood_synthetic_val(dataset_config, test_epoch, seen_classes, known_unseen_classes, dataset = data, threshold = j)
        result = model_train_obj.testing_split_ood_synthetic_seed(dataset_config, test_epoch, seen_classes, known_unseen_classes, dataset = data, threshold = threshold)

        # CVAE Writer
        write_path = os.path.join(dataset_config['write_path'], dataset_config['test_file_name']+'_seed{}'.format(args.seed))
        
        with open('{}_top1.csv'.format(write_path), mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([tpr, i, result['unseen_top1'], result['ood_top1']])

        