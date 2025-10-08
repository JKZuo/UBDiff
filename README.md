#  UB-Diff

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)
![Code](https://img.shields.io/badge/Code-python-purple)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Bridging User Dynamic Preferences: A Unified Bridge-based Diffusion Model for Next POI Recommendation**

In this paper, we propose a Unified Bridge-based Diffusion model (UB-Diff) for the Next POI Recommendation. 
UB-Diff directly bridges the transition path between the user’s historical visit distribution and the target distribution without relying on traditional Gaussian priors, significantly improving the accuracy of user dynamic interest distribution modeling.

-  We design a **Direction-aware POI Transition Graph Learning Module**. This module aggregates the attribute features on edges and jointly models temporal, spatial, and directional features, thereby providing a comprehensive representation of user behavior dependencies.

-  A **Novel Bridge-based Diffusion POI Generative Model** is proposed, which naturally connects the transformation between any two arbitrary distributions. This enables us to replace the Gaussian prior with the user’s historical interest distribution, effectively modeling the uncertainty of visits and adapting to changes in user interests.

-  We propose a **Unified Noise and Sampling Scheduling** for POI recommendation, which introduces a novel intermediate function that enables precise control over noise and modular optimization, enhancing the flexibility of the diffusion process.

The overall framework of our proposed UB-Diff model is illustrated in **Figure 1**.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/UBDiff/blob/main/fig.png" width="750"/>
</p>
<p align = "center">
<b>Figure 1. The overall framework of the proposed Unified Bridge-based Diffusion model (UB-Diff). </b> 
</p>

## Requirements

The code has been tested running under Python 3.8.

```shell
conda create --name UB-Diff python = 3.8

conda activate UB-Diff 

pip install -r requirements.txt 
```

## Data
Due to limitations of large datasets (the data file uploaded by GitHub cannot be larger than ***25MB***), you can download datasets through this Baidu Cloud link:

link: [DATA](https://pan.baidu.com/s/19YYwOL3YbzSszyk9G9tYOQ?pwd=diff ) 

password: diff

This folder (data/processed) contains 4 datasets, including

(1) **IST** (Istanbul in Turkey); 

(2) **TKY** (Tokyo in Japan); 

(4) **NYC** (New York City in USA); 

(5) **LA** (Los Angeles in USA).

We also provided the raw files at (data/raw).

All datasets are sourced from https://sites.google.com/site/yangdingqi/home/foursquare-dataset

where "5. Global-scale Check-in Dataset with User Social Networks" includes long-term (about 22 months from Apr. 2012 to Jan. 2014) global-scale check-in data collected from Foursquare. The check-in dataset contains 22,809,624 check-ins by 114,324 users on 3,820,891 venues.


## Running

Please modify the datasets in your path: DATA_PATH = '../UB-Diff/data/processed' in the **[gol.py]** file

You can use the small-scale LA dataset as an example to run it as：

```shell
python main.py --dataset LA --gpu 0 --dp 0.4
```

or

```shell
nohup python main.py --dataset LA  --gpu 0 > result_LA.log  2>&1 &
```

We employ an early stopping strategy during training, where the process is terminated if the performance on the validation set does not improve for 10 consecutive epochs, with the maximum number of training epochs set to 100.

## Cite
If you feel that this work has been helpful for your research, please cite it as: 

- J. Zuo, Z. Yao and Y. Zhang, "Bridging User Dynamic Preferences: A Unified Bridge-based Diffusion Model for Next POI Recommendation," in IEEE Transactions on Big Data, Early Access, doi: https://doi.org/10.1109/TBDATA.2025.3618453.

or

```tex
@ARTICLE{UB-Diff,
  author={Zuo, Jiankai and Yao, Zihao and Zhang, Yaying},
  journal={IEEE Transactions on Big Data}, 
  title={Bridging User Dynamic Preferences: A Unified Bridge-based Diffusion Model for Next POI Recommendation}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={POI recommendation;Diffusion model;Graph neural network;Self-attention;Location-based social networks},
  doi={10.1109/TBDATA.2025.3618453}}

```

keywords: POI Recommendation; Diffusion Model; Graph Neural Network; Self-attention; Location-based Social Networks (LBSNs).
