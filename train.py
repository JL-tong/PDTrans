import argparse
import logging
import os

import numpy as np
import math, copy, time
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from IPython import embed
from itertools import chain
import torch.nn.functional as functional
import utils
from utils import EarlyStopping
import PDTransformer as transformer
from evaluate import *
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast , GradScaler
import random
from opt import OpenAIAdam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger('PDTrans.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='PDTrans/data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='PDTrans/output_elect', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None, help='Optional, name of the file in --model_dir containing weights to reload before training')

parser.add_argument('--beta', type=float, default=10, help='hyperparameter of loss function')
parser.add_argument('--gamma', type=float, default=10, help='hyperparameter of loss function')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model,autoencoder,
        train_loader,
        optimizer,
        scaler,
        params: utils.Params,
        epoch) -> None:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    autoencoder.train()
    loss_epoch = np.zeros(len(train_loader))
    train_loss = []

    for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
        batch_size = train_batch.shape[0]
        train_batch = train_batch.to(torch.float32).to(params.device)  
        labels_batch = labels_batch.to(torch.float32).to(params.device)  
        idx = idx.unsqueeze(-1).to(params.device) 




        mu=None; logvar=None
        distribution_mu, distribution_sigma = model.forward(train_batch, idx, mu, logvar)  

        Recon_input = train_batch[:,1:params.predict_start+1,0]
        recon_x, recon_y, recon_z, mu, logvar = autoencoder(Recon_input,distribution_mu)
        loss_re = args.beta*transformer.kl_loss( mu, logvar) + args.gamma*transformer.loss_fn(recon_x, distribution_sigma, labels_batch, params.predict_start)
        # loss_r.append(loss_re.item())


        optimizer.zero_grad()
        loss =  loss_re + transformer.loss_fn(distribution_mu, distribution_sigma, labels_batch, params.predict_start)
        
        cuda_exist = torch.cuda.is_available()
        if cuda_exist:
            with autocast():
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # g_loss = loss.item() / params.predict_steps
        g_loss = loss.item() / params.train_window
        loss_epoch[i] = g_loss
        train_loss.append(loss.item())

        
        if i % 500 == 0:
            logger.info("R_loss: {} ; loss: {}".format(loss_re, loss))

    return np.average(train_loss),loss_epoch


def train_and_evaluate(model,autoencoder,
                       train_loader,valid_loader,test_loader: DataLoader,
                       optimizer,scheduler,
                       scaler,
                       params: utils.Params,
                       restore_file: str = None) -> None:

    '''Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    early_stopping = EarlyStopping(patience=100, verbose=True)

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    
    logger.info('begin training and evaluation')
    best_valid_q50 = float('inf')
    best_valid_q90 = float('inf')
    best_MAPE = float('inf')
    train_len = len(train_loader) 
    valid_len = len(valid_loader)


    q50_valid = np.zeros(params.num_epochs)
    q90_valid = np.zeros(params.num_epochs)
    MAPE_valid = np.zeros(params.num_epochs)

    loss_train_epoch = np.zeros((train_len * params.num_epochs))
    loss_valid = np.zeros((valid_len * params.num_epochs))

    valid_loss = []
    logger.info("PDTransformer have {} paramerters in total".format(sum(x.numel() for x in model.parameters()) + sum(x.numel() for x in autoencoder.parameters())))


    Avg_loss_train = np.zeros(params.num_epochs)
    Avg_loss_valid = np.zeros(params.num_epochs)

    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        # valid_metrics = evaluate(model, valid_loader, params, epoch)

        train_loss,loss_train_epoch[epoch * train_len:(epoch + 1) * train_len] = \
            train(model,autoencoder,train_loader,optimizer, \
                scaler,params,epoch)
        logger.info('NLL-loss:{}'.format(train_loss))
        if epoch<20:
            scheduler.step()
        logger.info('Learning rate:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))  
        
        valid_metrics = evaluate(model,autoencoder, valid_loader, params, epoch)
        test_metrics = evaluate(model,autoencoder, test_loader, params, epoch)
        loss_valid[epoch * valid_len:(epoch + 1) * valid_len] = valid_metrics['test_loss']

        Avg_loss_train[epoch] = train_loss
        Avg_loss_valid[epoch] = valid_metrics['test_loss']

        q50_valid[epoch] = valid_metrics['q50']
        q90_valid[epoch] = valid_metrics['q90']
        # MAPE_valid[epoch] = valid_metrics['MAPE']

        valid_loss.append(valid_metrics['q50'])
        
        is_best = q50_valid[epoch] <= best_valid_q50

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'auto_dict': autoencoder.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                                epoch=epoch,
                                is_best=is_best,
                                checkpoint=params.model_dir)

        if is_best:
            logger.info('- Found new best Q90/Q50')
            best_valid_q50 = q50_valid[epoch]
            best_json_path = os.path.join(params.model_dir, 'metrics_valid_best_weights.json')
            utils.save_dict_to_json(valid_metrics, best_json_path)
            best_valid_ND = epoch

        utils.save_loss(loss_train_epoch[epoch * train_len:(epoch + 1) * train_len], args.dataset + '_' + str(epoch) +'-th_epoch_loss', params.plot_dir)
        utils.save_loss(loss_valid[epoch * valid_len:(epoch + 1) * valid_len], args.dataset + '_' + str(epoch) +'-th_epoch_valid_loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir, 'metrics_valid_last_weights.json')
        utils.save_dict_to_json(valid_metrics, last_json_path)
        early_stopping(valid_loss[-1], model)
        if early_stopping.early_stop:
            print("Early stopping")
            # save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'auto_dict': autoencoder.state_dict(),
                                'optim_dict': optimizer.state_dict()},
                                filepath=params.model_dir)
            break
    

    loss_train_epoch = np.array(loss_train_epoch)
    plt.figure()
    plt.figure(figsize=(20, 10))
    plt.plot(Avg_loss_train[1:], color= 'red',label='loss')
    # plt.plot(Avg_loss_valid[1:], color= 'blue',label='loss')
    plt.xlabel('Epoch')
    plt.savefig(params.model_dir+'/DeFigs/Loss.png')
    plt.close()
    


    if args.save_best:
        f = open('./param_search.txt', 'w')
        f.write('-----------\n')
        list_of_params = list(params.__dict__.keys())
        print_params = ''
        for param in list_of_params:
            param_value = getattr(params, param)
            print_params += f'{param}: {param_value:.2f}'
        print_params = print_params[:-1]
        f.write(print_params + '\n')
        f.write('Best ND: ' + str(best_valid_ND) + '\n')
        logger.info(print_params)
        logger.info(f'Best ND: {best_valid_ND}')
        f.close()

if __name__ == '__main__':
    # setup_seed(12)
    # Load the parameters from json file
    args = parser.parse_args()

    model_dir = args.model_name
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)
    params.relative_metrics = args.relative_metrics
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.dataset = args.dataset

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass
    
    # use GPU if available
    params.ngpu = torch.cuda.device_count()

    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info('Using Cuda...')
    c = copy.deepcopy

    attn = transformer.MultiHeadedAttention(params)
    ff = transformer.PositionwiseFeedForward(params.d_model, d_ff=params.d_ff, dropout=params.dropout)
    position = transformer.PositionalEncoding(params.d_model, dropout=params.dropout)
    emb = transformer.Embedding(params, position)
    ############################################################
    encoder  = transformer.Encoder(params, transformer.EncoderLayer(params, c(attn), c(ff), dropout=params.dropout))
    decoder = transformer.Decoder(params, transformer.DecoderLayer(params, c(attn), c(attn), c(ff), dropout=params.dropout))
    generator = transformer.Generator(params)
    model = transformer.EncoderDecoder(params= params, emb = emb, encoder = encoder, decoder = decoder, generator = generator)
    autoencoder = transformer.AutoEncoder(params)
    autoencoder.to(params.device)


    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     discriminator = nn.DataParallel(discriminator)

    model.to(params.device)



    utils.set_logger(os.path.join(model_dir, 'train.log'))

    logger.info('Loading the datasets...')


    train_set = TrainDataset(data_dir, args.dataset, params.num_class)  
    valid_set = ValidDataset(data_dir, args.dataset, params.num_class)
    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=RandomSampler(train_set), num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=params.predict_batch, sampler=RandomSampler(valid_set), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('Loading complete.')

    n_updates_total = (train_set.__len__() // params.batch_size) * params.num_epochs


    optimizer = optim.Adam(chain(model.parameters(),autoencoder.parameters()), lr = args.lr)

    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma = 0.8)
 
    scaler = GradScaler()
    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    train_and_evaluate(model,autoencoder,
                       train_loader,valid_loader,test_loader,
                       optimizer,scheduler,
                       scaler,
                       params,
                       args.restore_file)
