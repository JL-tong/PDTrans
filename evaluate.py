from IPython import embed
import matplotlib.pyplot as plt
import argparse
import logging
import os

import copy
import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
#import model.net as net
import PDTransformer as transformer
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import properscoring as ps



logger = logging.getLogger('Transformer.Eval')
#torch.cuda.set_device(2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='M4', help='Name of the dataset')
# parser.add_argument('--dataset', default='exchange_rate', help='Name of the dataset')
# parser.add_argument('--dataset', default='traffic', help='Name of the dataset')
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
# parser.add_argument('--dataset', default='Solar', help='Name of the dataset')
parser.add_argument('--data-folder', default='data',help='Parent dir of the dataset')
# parser.add_argument('--model-name', default='output_exchange',help='Directory containing params.json')
# parser.add_argument('--model-name', default='output_M4',help='Directory containing params.json')
# parser.add_argument('--model-name', default='output_traffic',help='Directory containing params.json')
parser.add_argument('--model-name', default='output_elect',help='Directory containing params.json')
# parser.add_argument('--model-name', default='output_Solar',help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true',
                    help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true',
                    help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def evaluate(model,autoencoder, test_loader, params, plot_num=None):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    data_Norm = np.load('PDTrans/data/'+args.dataset+'/data_Norm.npy')
    CRPS = 0
    

    model.eval()
    with torch.no_grad():
        # plot_batch = np.random.randint(len(test_loader)-1)
        summary_metric = {}
        raw_metrics = utils.init_metrics()
        sum_mu = torch.zeros([740, params.predict_steps]).to(params.device)
        sum_sigma = torch.zeros([740, params.predict_steps]).to(params.device)
        true = torch.zeros([740, params.predict_steps]).to(params.device)


        for i, (test_batch, id_batch, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.to(torch.float32).to(params.device)
            NormData = torch.tensor(data_Norm[id_batch,:]).to(torch.float32).to(params.device)
            id_batch = id_batch.unsqueeze(-1).to(params.device)
            labels = labels.to(torch.float32).to(params.device)
            batch_size = test_batch.shape[0]
            Sigma = torch.zeros(batch_size, labels.shape[1], device=params.device)  # scaled
            input_mu = torch.zeros(batch_size, params.predict_start, device=params.device)  # scaled


            mu=None; logvar=None
            distribution_mu, distribution_sigma= transformer.test( model, autoencoder,params, test_batch,id_batch,mu, logvar)
            Recon_input = test_batch[:,1:-params.predict_steps+1,0]
            distribution_mu, recon_y, recon_z, mu, logvar = autoencoder(Recon_input,distribution_mu)


            scale_o = NormData[:,1].unsqueeze(-1) ; mean_o = NormData[:,0].unsqueeze(-1)
            sample_mu = scale_o * distribution_mu + mean_o  #anti-z-score
            sigma= scale_o * distribution_sigma
            Sigma[:,params.test_predict_start:] = scale_o * distribution_sigma
            labels = scale_o * labels + mean_o
            #Q90
            gaussian = torch.distributions.normal.Normal(sample_mu, sigma)
            sample_90 = gaussian.icdf(torch.tensor(0.9))

            raw_metrics = utils.update_metrics(raw_metrics, sample_mu, sigma, sample_90, labels, params.predict_start,
                                               relative=params.relative_metrics, sample=True)
            


    
            recon_y = scale_o * recon_y + mean_o
            recon_z = scale_o * recon_z + mean_o
            plot_decomposition(params,sample_mu,recon_y,recon_z,labels,i,epoch=None)



            plot_mu = torch.cat((input_mu, sample_mu), axis=1)


            # plot_eight_windows('../PDTrans',
            # plot_mu.to('cpu').detach().numpy(),recon_y.to('cpu').detach().numpy(),recon_z.to('cpu').detach().numpy(),
            # Sigma.to('cpu').detach().numpy(),labels.to('cpu').detach().numpy(),50,30,4,raw_metrics)

   
            
            # if(i==0):
            #     sum_mu = sample_mu 
            #     # sum_q90= sample_q90
            #     true = labels[:, -params.predict_steps:]
            # else:
            #     sum_mu = torch.cat([sum_mu, sample_mu], 0)
            #     # sum_q90 = torch.cat([sum_q90, sample_q90], 0)
            #     true = torch.cat([true, labels[:, -params.predict_steps:]], 0)
            
        summary_metric = utils.final_metrics(raw_metrics,sample=True)
        

        metrics_string = '; '.join('{}: {:05.3f}'.format(k, v)
                                   for k, v in summary_metric.items())

        logger.info('- Full test metrics: ' + metrics_string)
        print('CRPS:', CRPS/(distribution_mu.shape[1]*len(test_loader)))
    return summary_metric


def plot_decomposition(params,pred,recon_y,recon_z,labels,batch,epoch=None):
    import random
    import pandas as pd
    import matplotlib.pyplot as plt

    rd = random.randint(0,pred.shape[0]-1)
    pred= pred[rd,:].to('cpu').detach().numpy()
    recon_y= recon_y[rd,:].to('cpu').detach().numpy()
    recon_z= recon_z[rd,:].to('cpu').detach().numpy()
    Labels = labels[rd,:].to('cpu').detach().numpy()

    T = np.array(range(labels[rd,:].__len__()))
    plt.figure()
    plt.figure(figsize=(20, 10))
    plt.plot(T,Labels, label='Label')
    plt.plot(T[-params.predict_steps:],pred, label='pred')
    plt.xlabel('time')
    if epoch is None:
        plt.savefig(params.model_dir+'/DeFigs/pred{}.png'.format(batch))
    else:
        plt.savefig(params.model_dir+'/DeFigs/pred{}-{}.png'.format(epoch,batch))
    plt.close()

    plt.figure()
    plt.figure(figsize=(20, 10))
    plt.plot(pred, label='pred',color='k')
    plt.plot(recon_y, label='recon_y',color='r')
    plt.plot(recon_z, label='recon_z',color='b')
    plt.xlabel('time')
    if epoch is None:
        plt.savefig(params.model_dir+'/DeFigs/subpred{}.png'.format(batch))
    else:
        plt.savefig(params.model_dir+'/DeFigs/subpred{}-{}.png'.format(epoch,batch))
    plt.close()



def plot_eight_windows(plot_dir,
                       predict_values,recon_y,recon_z,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):

    x = np.arange(window_size)
    # f = plt.figure(figsize=(8, 84), constrained_layout=True)
    f = plt.figure(figsize=(4, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)
    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x[predict_start:], predict_values[m,predict_start:], color='b')
        ax[k].plot(x[predict_start:], recon_y[m], color='y', label='trend')
        ax[k].plot(x[predict_start:], recon_z[m], color='g', label='seasonality')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                           predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                           alpha=0.2)
        # ax[k].plot(x[predict_start:], labels[m, predict_start:], color='r')
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        #metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})

        # plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
        #     f'RMSE: {plot_metrics["RMSE"][m]: .3f}' \
        #     f'Q50: {plot_metrics["Q50"][m]:.3f}' \
        #     f'Q90: {plot_metrics["Q90"][m]:.3f}'
        # if sampling:
        #     plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
        #                         f'rou50: {plot_metrics["rou50"][m]: .3f}'
        # ax[k].set_title(plot_metrics_str, fontsize=10)

    # f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'),format='png')
    f.savefig(os.path.join(plot_dir, str(plot_num) + '.svg'),format='svg')
    plt.close()



if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    model_dir = args.model_name
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(
        json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')

    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary

    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.cuda.manual_seed(240)
    logger.info('Using Cuda...')

    c = copy.deepcopy
    attn = transformer.MultiHeadedAttention(params)
    ff = transformer.PositionwiseFeedForward(params.d_model, d_ff=params.d_ff, dropout=params.dropout)
    position = transformer.PositionalEncoding(params.d_model, dropout=params.dropout)
    ge = transformer.Generator(params)
    emb = transformer.Embedding(params, c(position))
    
    model = transformer.EncoderDecoder(params= params, emb = emb, encoder = transformer.Encoder(params, transformer.EncoderLayer(params, c(attn), c(ff), dropout=params.dropout)), decoder = transformer.Decoder(params, transformer.DecoderLayer(params, c(attn), c(attn), c(ff), dropout=params.dropout)), generator = ge)
   
    model.to(params.device)

    autoencoder = transformer.AutoEncoder(params)
    autoencoder.to(params.device)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch,
                             sampler=RandomSampler(test_set), num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = transformer.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        model_dir, args.restore_file + '.pth.tar'), model,autoencoder)

    test_metrics = evaluate(model, autoencoder, test_loader,
                            params, params.sampling)
    save_path = os.path.join(
        model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)


