# GeoExplainer

## Introduction
This repository holds the codebase, dataset and models for the paper:

**GeoExplainer: Interpreting Graph Convolutional Networks with geometric masking**. 
Rui Yu, Yanshan Li, Huajie Liang, Zhiyuan Chen,
Neurocomputing,
Volume 605,
2024,
128393,
ISSN 0925-2312,
https://doi.org/10.1016/j.neucom.2024.128393.
[[Paper]](https://www.sciencedirect.com/science/article/pii/S0925231224011640)

**We use GeoExplainer to interpret the STGCN pre-trained models.**

ST-GCN is able to exploit local pattern and correlation from human skeletons. ST-GCN：
**Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)


## Prerequisites
- Python3 (>3.5)
- [PyTorch](http://pytorch.org/)
- [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) **with** [Python API](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#python-api). (Optional: for demo only)
- Other Python libraries can be installed by `pip install -r requirements.txt`
<!-- - FFmpeg (Optional: for demo only), which can be installed by `sudo apt-get install ffmpeg` -->


### Installation
``` shell
cd GeoExplainer;
cd torchlight; python setup.py install; cd ..
```

## Data Preparation

We experimented on skeleton-based action recognition datast: and **NTU RGB+D**.
Before training and testing, for convenience of fast data loading,
the datasets should be converted to proper file structure. 
You can download the pre-processed data from 
[GoogleDrive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb)
and extract files with
``` 
cd st-gcn
unzip <path to st-gcn-processed-data.zip>
```
After uncompressing, rebuild the database by this command.

#### NTU RGB+D
NTU RGB+D can be downloaded from [their website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).
Only the **3D skeletons**(5.8GB) modality is required in our experiments. After that, this command should be used to build the screw database for training or evaluation:
```
python tools/ntu_gendata_screw.py --data_path <path to nturgbd+d_skeletons>
```
where the ```<path to nturgbd+d_skeletons>``` points to the 3D skeletons modality of NTU RGB+D dataset you download.


## Testing Pretrained Models
<!-- ### Evaluation
Once datasets ready, we can start the evaluation. -->

To evaluate ST-GCN model pretrained on **NTU RGB+D**, 

For **cross-view** evaluation in **NTU RGB+D**, run
```
python main.py recognition -c config/st_gcn/ntu-xview/test.yaml
```
For **cross-subject** evaluation in **NTU RGB+D**, run
```
python main.py recognition -c config/st_gcn/ntu-xsub/test.yaml
``` 

<!-- Similary, the configuration file for testing baseline models can be found under the ```./config/baseline```. -->

To speed up evaluation by multi-gpu inference or modify batch size for reducing the memory cost, set ```--test_batch_size``` and ```--device``` like:
```
python main.py recognition -c <config file> --test_batch_size <batch size> --device <gpu0> <gpu1> ...
```

### Explaining

For explaining **cross-view**, run
```
python main.py ExplainerScrew_frame_sparsity_forgithub -c config/st_gcn/ntu-xview/test.yaml
```
For explaining **cross-subject**, run
```
python main.py ExplainerScrew_frame_sparsity_forgithub -c config/st_gcn/ntu-xview/test.yaml
```
Three evaluation metrics are Fidelity^{acc},Fidelity^{prob}, and Sparsity. 


## Citation
Please cite the following paper if you use this repository in your reseach.
```
@article{YU2024128393,
title = {GeoExplainer: Interpreting Graph Convolutional Networks with geometric masking},
journal = {Neurocomputing},
volume = {605},
pages = {128393},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.128393},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224011640},
author = {Rui Yu and Yanshan Li and Huajie Liang and Zhiyuan Chen},
}

@inproceedings{stgcn2018aaai,
  title     = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author    = {Sijie Yan and Yuanjun Xiong and Dahua Lin},
  booktitle = {AAAI},
  year      = {2018},
}
```
