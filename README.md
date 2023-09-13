# PDTrans (SDM 2023)
This repository provides an implementation for PDTrans as described in the paper:

> Probabilistic Decomposition Transformer for Time Series Forecasting.
> Junlong Tong, Liping Xie, Kanjian Zhang.
> SDM, 2023.
> [[Paper]](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch54)


## Probabilistic Decomposition Transformer
<p align="justify">
Time series forecasting is crucial for many fields, such as disaster warning, weather prediction, and energy consumption. The Transformer-based models are considered to have revolutionized the field of time series. However, the autoregressive form of the Transformer introduces cumulative errors in the inference stage. Furthermore, the complex temporal pattern of the time series leads to an increased difficulty for the models in mining reliable temporal dependencies. In this paper, we propose the Probabilistic Decomposition Transformer model, which provides a flexible framework for hierarchical and decomposable forecasts. The hierarchical mechanism utilizes the forecasting results of Transformer as conditional information for the generative model, performing sequence-level forecasts to approximate the ground truth, which can mitigate the cumulative error of the autoregressive Transformer. In addition, the conditional generative model encodes historical and predictive information into the latent space and reconstructs typical patterns from the latent space, such as seasonality and trend terms. The process provides a flexible framework for the separation of complex patterns through the interaction of information in the latent space. Extensive experiments on several datasets demonstrate the effectiveness and robustness of the model, indicating that it compares favorably with the state-of-the-art.</p>

* Architecture
<p align="center">
<img src=".\PDTrans.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall architecture of PDTrans.
</p>

* Loss function

$$ \mathcal{L}= \gamma \mathcal{L}_{NLL}+\beta \mathcal{L}_{KL} +  \mathcal{L}_{R}.$$
$$ \mathcal{L}_{NLL} = \sum_{t} l\left(Y_{t} \mid \mu_{t}, \sigma_{t}\right).$$
$$\mathcal{L}_{KL} = D_{K L}\left(q_{\phi}\left(z \mid Y_{1: t_{0}}, \mu_{t_{0}+1: t_{0}+\tau}\right) \| p_{\theta}\left(z \mid Y_{1: t_{0}}\right)\right). $$
$$\mathcal{L}_{R} =  \sum_{t} l'\left(\hat{Y}_{t} \mid \hat{\mu}_{t}, \hat{\sigma}_{t}\right).$$
## Requirements
* Python 3.8
* PyTorch 1.8

## Data
  * Electricity dataset: http://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
  * Traffic dataset: http://archive.ics.uci.edu/dataset/204/pems+sf
  * Solar dataset: https://www.nrel.gov/grid/solar-power-data.html
  * M4 dataset: https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset
  * Exchange dataset: https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate
## Usage
1. Preprocess the data:
   ```bash
   python prepdata.py
   ```
2. Restore the saved model and make prediction:
   ```bash
   python evaluate.py --dataset='elect' --model-name='output_elect' --restore-file='best'
   ```
3. Train the model:
   ```bash
   python train.py --dataset='elect' --model-name='output_elect' 
   ```
## Reproducibility
* To easily reproduce the results, we provide the experiment script on electricity dataset. You can reproduce the experiment results by:
   ```bash
   bash ./script/PDTrans_elect.sh
   ```
## Citation 
If you find this repository useful, please cite our paper.
```
@inproceedings{tong2023probabilistic,
  title={Probabilistic decomposition transformer for time series forecasting},
  author={Tong, Junlong and Xie, Liping and Zhang, Kanjian},
  booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
  pages={478--486},
  year={2023},
  organization={SIAM}
}
```

## Contact
If you have any questions, please contact: [jl-tong@sjtu.edu.cn](jl-tong@sjtu.edu.cn)