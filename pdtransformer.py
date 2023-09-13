'''Defines the neural network, loss function and metrics'''
import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils
from torch.nn.utils import weight_norm
import logging
from torch.autograd import grad
from typing import Tuple
from IPython import embed
import warnings


warnings.filterwarnings("ignore")

logger = logging.getLogger('Transformer.Net')

'''AutoEncoder'''
class VAEEncoderLayer(nn.Module):
    def __init__(self, params):
        super(VAEEncoderLayer, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(params.inter_dim, params.inter_dim),
            nn.LeakyReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(params.inter_dim, params.inter_dim),
            nn.LeakyReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(params.inter_dim, params.inter_dim),
            nn.LeakyReLU(),
        )
        self.encoder1 = nn.Sequential(
            nn.Linear(params.inputLen,params.inter_dim//2),
            nn.LeakyReLU(),
        ) 
        self.encoder1_pred = nn.Sequential(
            nn.Linear(params.input_dim, params.inter_dim//2),
            nn.LeakyReLU(),
        ) 
        self.encoder2 = nn.Linear(params.inter_dim, params.latent_dim * 2)
        self.init_weights()

    def init_weights(self):
        self.encoder1_pred[0].weight.data.normal_(0, 0.01)
        self.encoder1[0].weight.data.normal_(0, 0.01)
        self.encoder2.weight.data.normal_(0, 0.01)
        self.linear1[0].weight.data.normal_(0, 0.01)
        self.linear2[0].weight.data.normal_(0, 0.01)
        self.linear3[0].weight.data.normal_(0, 0.01)

    def forward(self, his_seq, pred_seq):
        his_seq = self.encoder1(his_seq.unsqueeze(1))
        pred_seq = self.encoder1_pred(pred_seq.unsqueeze(1))
        enc =  torch.cat([his_seq,pred_seq],-1)
        enc = enc + self.linear1(enc)
        enc = enc + self.linear2(enc)
        enc = enc + self.linear3(enc)
        enc = self.encoder2(enc)
        return enc

class VAEDecoderLayer(nn.Module):
    def __init__(self, params):
        super(VAEDecoderLayer, self).__init__()
        self.decoder1 = nn.Sequential(
            nn.Linear(params.latent_dim, params.inter_dim//2),
            nn.LeakyReLU(),
        )

        self.decoder1y = nn.Sequential(
            nn.Linear(params.inputLen, params.inter_dim//2),
            nn.LeakyReLU(),
        )


        self.linear1 = nn.Sequential(
            nn.Linear(params.inter_dim, params.inter_dim),
            nn.LeakyReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(params.inter_dim, params.inter_dim),
            nn.LeakyReLU(),
        )
        self.decoder2 = nn.Linear(params.inter_dim, params.input_dim)

        self.CausalConv1d1 = nn.Sequential(
            CausalConv1d(1, 1, kernel_size=5),
            nn.LeakyReLU(),
        )
        self.CausalConv1d2 = nn.Sequential(
            CausalConv1d(1, 1, kernel_size=5),
            nn.LeakyReLU(),
        )
        self.CausalConv1d3 = nn.Sequential(
            CausalConv1d(1, 1, kernel_size=5),
            nn.LeakyReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Linear(params.latent_dim, params.inter_dim//2),
            nn.LeakyReLU(),
        )
        self.decoder3y = nn.Sequential(
            nn.Linear(params.inputLen, params.inter_dim//2),
            nn.LeakyReLU(),
        )
        self.decoder4 = nn.Linear(params.inter_dim, params.input_dim)

        self.LinearDecoder = nn.Sequential(
            nn.Linear(params.inter_dim, params.inter_dim),
            nn.LeakyReLU(),
        )

        self.init_weights()

    def init_weights(self):
        self.linear1[0].weight.data.normal_(0, 0.01)
        self.linear2[0].weight.data.normal_(0, 0.01)
        self.decoder1[0].weight.data.normal_(0, 0.01)
        self.decoder2.weight.data.normal_(0, 0.01)
        self.decoder4.weight.data.normal_(0, 0.01)
        self.decoder3[0].weight.data.normal_(0, 0.01)
        self.decoder1y[0].weight.data.normal_(0, 0.01)
        self.decoder3y[0].weight.data.normal_(0, 0.01)

    def forward(self, hidden_z, his_seq):
        z_season = self.decoder1(hidden_z)
        h_season = self.decoder1y(his_seq)
        season =  torch.cat([z_season,h_season],-1)
        season = season + self.linear1(season)
        season = season + self.linear2(season)
        recon_season = self.decoder2(season)
 
        z_trend = self.decoder3(hidden_z)
        h_trend = self.decoder3y(his_seq)
        trend =  torch.cat([z_trend,h_trend],-1)
        trend = self.CausalConv1d1(trend.unsqueeze(1)).squeeze(1)
        trend = trend + self.CausalConv1d2(trend.unsqueeze(1)).squeeze(1)
        trend = trend + self.CausalConv1d3(trend.unsqueeze(1)).squeeze(1)
        recon_trend = self.decoder4(trend)

        recon_seq = recon_trend + recon_season
        return recon_seq, recon_trend, recon_season


class AutoEncoder(nn.Module):
    def __init__(self, params):
        super(AutoEncoder, self).__init__()
        self.encoder = VAEEncoderLayer(params)
        self.decoder = VAEDecoderLayer(params)  

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, his_seq, pred_seq):
        org_size = his_seq.size()
        batch = org_size[0]
        his_seq = his_seq.view(batch, -1)

        enc = self.encoder(his_seq, pred_seq)

        mu, logvar = enc.chunk(2, dim=-1)
        hidden_z = self.reparameterise(mu.squeeze(1), logvar.squeeze(1))


        recon_seq, recon_trend, recon_season = self.decoder(hidden_z, his_seq)

        return recon_seq, recon_trend, recon_season, mu, logvar



'''Generator'''       
class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        # self.q50 = nn.Linear(params.d_model, 1)
        # self.q90 = nn.Linear(params.d_model, 1)
        self.distribution_mu = nn.Sequential(
            # nn.LayerNorm(params.d_model+params.latent_dim, eps=1e-6),
            # # nn.Linear(params.d_model+params.latent_dim, params.latent_dim),
            # nn.Linear(params.d_model+params.latent_dim, 1),
            nn.LayerNorm(params.d_model, eps=1e-6),
            nn.Linear(params.d_model, 1),
        )
        self.distribution_sigma = nn.Sequential(
            # nn.LayerNorm(params.d_model+params.latent_dim, eps=1e-6),
            # # nn.Linear(params.d_model+params.latent_dim, params.latent_dim),
            # nn.Linear(params.d_model+params.latent_dim, 1),
            # nn.Softplus(),
            nn.LayerNorm(params.d_model, eps=1e-6),
            nn.Linear(params.d_model, 1),
            nn.Softplus(),
        )
        # self.embedding = nn.Embedding(params.latent_dim, 5)
        self.para = params
        # self.emdedding = nn.Embedding(params.laten_dim, params.d_model)

    def forward(self, x, mu, logvar):
        
        # mu = mu.repeat(1,x.shape[1],1)
        # logvar = logvar.repeat(1,x.shape[1],1)

        # # x_cat = torch.cat((x,onehot_embed),-1)
        # mu_cat = torch.cat((x,mu),-1)
        # sigma_cat = torch.cat((x,logvar.exp()),-1)
        # distribution_mu = self.distribution_mu(mu_cat)
        # distribution_sigma = self.distribution_sigma(sigma_cat)

        distribution_mu = self.distribution_mu(x)
        distribution_sigma = self.distribution_sigma(x)
        return  distribution_mu.squeeze(-1), distribution_sigma.squeeze(-1)


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=0,dilation=dilation,groups=groups,bias=bias)
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):#(Batch, input_channel, seq_len)
        pad = (self.__padding-self.__padding//2, self.__padding//2)
        inputs = F.pad(input, pad)  
        return super(CausalConv1d, self).forward(inputs)



'''Transformer EncoderDecoder'''
class EncoderDecoder(nn.Module):
    def __init__(self, params, emb, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.emb = emb  #class-Embedding
        self.encoder = encoder     #class-Encoder --EncoderLayer
        self.decoder = decoder   #class-Decoder
        self.generator = generator #class-Genarator
        self.predict_steps = params.predict_steps  


    ### Encoder: [0,y1~y167]; Decoder: [y168~y191]_Mask   ------>    [y169~y192]
    def forward(self, x, idx, mu, logvar):
        #x[:,:-self.predict_steps,:]
        src_mask, encoder_out = self.encode(x[:,:-self.predict_steps,:], idx)  #[0,y1~y167]
        #mu_en, sigma_en = self.generator(encoder_out)
        decoder_out = self.decode(encoder_out, x[:,-self.predict_steps:,:], idx, src_mask) 
        distribution_mu, distribution_sigma = self.generator(decoder_out, mu, logvar)
       
        #mu = torch.cat((mu_en, mu_de), 1)
        #sigma = torch.cat((sigma_en, sigma_de), 1)
        return distribution_mu, distribution_sigma


    def encode(self, x, idx):  #x[:,:-self.predict_steps,:]
        src_mask = (x[:,:,0]!=0).unsqueeze(-2) 
        embeded = self.emb(x, idx)  #outout=[x;idx_embedding]+PE
        encoder_out = self.encoder(embeded, None)
        
        return src_mask, encoder_out

    def decode(self, memory, x, idx, src_mask):    #memory = encoder_out  
        tgt_mask = make_std_mask(x[:,:,0], 0)  #mask the forecasting series 
        embeded = self.emb(x, idx)   #outout=[x;idx_embedding]+PE
        decoder_out = self.decoder(embeded, memory, None, tgt_mask)  #embeding + encoder_out + mask
        return decoder_out

def make_std_mask(tgt, pad):  
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
    # [ True, False, False,  ..., False, False, False],
    # [ True,  True, False,  ..., False, False, False],
    # [ True,  True,  True,  ..., False, False, False],
    # ...
    # [ True,  True,  True,  ...,  True, False, False],
    # [ True,  True,  True,  ...,  True,  True, False],
    # [ True,  True,  True,  ...,  True,  True,  True]
    
def subsequent_mask(size):   #sequence mask
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

'''Transformer Encoder'''
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, params, layer):  #layer: class-EncoderLayer
        super(Encoder, self).__init__()
        self.layers = clones(layer, params.N)   #duplicate N times (number of layers)
        self.norm = LayerNorm(layer.size)
         
    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, src_mask)   #caluculate self-attention based on x and mask
        encoder_output = self.norm(x)
        return encoder_output
        
class EncoderLayer(nn.Module):
#(1)Multi-head self-attention + Residual Connection +  Norm
#(2)Feed forward + Residual Connection + Norm
    def __init__(self, params, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.params = params
        self.self_attn = self_attn   #class - MultiHeadedAttention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(params.d_model, dropout), 2)  #residual connection
        self.size = params.d_model
    
    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) #Multi-head self-attention + ResCon &  Norm
        x = self.sublayer[1](x, self.feed_forward)  #Feed forward + ResCon &  Norm
        return x   

'''Transformer Decoder'''
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, params, layer):
        super(Decoder, self).__init__()
        self.layers = clones(layer, params.N) #duplicate N times (number of layers) #Attention is all your need. 
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask): 
        for layer in self.layers:  
            x = layer(x, memory, src_mask, tgt_mask)
        decoder_output = self.norm(x)
        return decoder_output

class DecoderLayer(nn.Module):
#(1)Masked multi-head self-attention + Residual Connection +  Norm
#(2)Multi-head self-attention + Residual Connection +  Norm
#(3)Feed forward + Residual Connection + Norm
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, params, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = params.d_model
        self.self_attn = self_attn    #Multi-head attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory    #memory = encoder_out ； x = embedding 
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) 
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))   
        src_mask=None
        x = self.sublayer[2](x, self.feed_forward)
        return x
     
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


'''Embedding'''
class Embedding(nn.Module):
    def __init__(self, params, position):
        super(Embedding, self).__init__()
        self.params = params
        ##torch.nn.Embedding(num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(params.num_class, params.embedding_dim) 
        '''
        if(params.dataset == "wind"):
            self.embed1 = nn.Linear(6, params.d_model)
        else:
        '''
        self.embedding
        # self.embed1 = nn.Linear(params.embedding_dim + params.cov_dim+ 3, params.d_model)
        self.embed1 = nn.Linear(params.embedding_dim + params.cov_dim+ 1, params.d_model)
        self.embed2 = position  #class-PositionalEncoding
        
    def forward(self, x, idx):
        "Pass the input (and mask) through each layer in turn.  x : [bs, window_len, 5] "

        idx = idx.repeat(1, x.shape[1]) # idx is the store id of this batch , [bs, window_len]
        '''
        if(self.params.dataset=="wind"):
            idx = torch.unsqueeze(idx, -1)
            output = torch.cat((x, idx.float()), dim=2) # [bs, widow_len, 25]  [bs, window]  wind dataset!!!
        else:
        '''
        onehot_embed = self.embedding(idx) #[bs, windows_len, embedding_dim(default 20)] 
        try:
            output = torch.cat((x, onehot_embed), dim=-1)  
            # output = self.embed2(self.embed1(output))
            output = self.embed1(output)  #embedding
            output = self.embed2(output)  #position encoding  ; outout=idx_embedding+PE
        except:
            embed()
        return output

   
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=500): # TODO:max_len
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
        
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



'''self-attention'''
def attention(query, key, value, params, mask=None, dropout=None, alpha=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    #operation MatMul and Scale
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        try:
            scores = scores.masked_fill(mask == 0, -1e9)   
        except:
            embed()
    #operation Softmax
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    p_attn = p_attn.to(torch.float32)
    #operation MatMul
    return torch.matmul(p_attn, value), scores, p_attn

class MultiHeadedAttention(nn.Module): 
    def __init__(self, params, dropout=0.2):  # TODO : h , dropout
        "Take in model size and number of heads." 
        super(MultiHeadedAttention, self).__init__()
        assert params.d_model % params.h == 0

        self.d_k = params.d_model // params.h
        self.h = params.h  #head
        self.linears = clones(nn.Linear(params.d_model, params.d_model), 4)   
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.params = params
        self.scores = None
        # self.alpha_choser = AlphaChooser(params.h)
        self.alpha = None
        self.attn_type = params.attn_type

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        #  d_model -> h*d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        # if self.attn_type=='entmax':
        #     self.alpha = self.alpha_choser()
        x, self.scores, self.attn = attention(query, key, value, self.params, mask=mask, 
                                     dropout=self.dropout, alpha=self.alpha)
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
       
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6): 
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



def test(model,autoencoder, params, x,  id_batch,mu, logvar):
    batch_size = x.shape[0]
    sample_mu = torch.zeros(batch_size, params.predict_steps, device=params.device)
    sample_q90 = torch.zeros(batch_size, params.predict_steps, device=params.device)
    src_mask, memory = model.encode(x[:, :params.predict_start,:], id_batch)
    for t in range(params.predict_steps):
        ys = x[:, params.predict_start:params.predict_start+t+1,:]
        out = model.decode(memory, ys, id_batch, src_mask)
        distribution_mu, distribution_sigma = model.generator(out,mu, logvar)
        if t < (params.predict_steps - 1):
            x[:, params.predict_start+t+1, 0] = distribution_mu[:, -1] 
    # return sample_mu, sample_q90
    return distribution_mu, distribution_sigma


def loss_fn(mu: Variable, sigma: Variable, labels: Variable, predict_start):
    labels = labels[:,predict_start:]
    zero_index = (labels != 0)
    mask = sigma == 0
    sigma_index = zero_index * (~mask)
    distribution = torch.distributions.normal.Normal(mu[sigma_index], sigma[sigma_index])
    likelihood = distribution.log_prob(labels[sigma_index])
    # likelihood = -(labels-mu)**2/(2*sigma**2)-torch.log(sigma)-0.5*math.log(2*math.pi)
    return -torch.mean(likelihood)

def kl_loss(mu: Variable, logvar: Variable):
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KL


# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]
    
def accuracy_MAPE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    numerator = 0
    denominator = 0
    #pred_samples = samples.shape[0]
    rou_pred = mu
    abs_diff = labels - rou_pred
    numerator += 2 * (torch.sum(rou * abs_diff[labels > rou_pred]) - torch.sum(
        (1 - rou) * abs_diff[labels <= rou_pred])).item()
    denominator += torch.sum(labels).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]

def MAPE(mu:torch.Tensor, labels:torch.Tensor):
    zero_index = (labels != 0)
    diff = mu[zero_index] -labels[zero_index]
    lo = torch.mean(torch.abs(diff / labels[zero_index])) *100
    return lo

def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    mask = labels == 0
    mu[mask] = 0.

    abs_diff = np.abs(labels - mu)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < mu] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= mu] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)
    
    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result
