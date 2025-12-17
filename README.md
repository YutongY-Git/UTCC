# UTCC-Mapping
This is the pytorch codes for our paper:"Accounting for Shadows in Urban Canopy Coverage Mapping: A Hierarchical Spatial-Context Framework using Google Earth Imagery".This project provides a deep learning workflow for accurately estimating Urban Tree Canopy Coverage (UTCC) in high-density urban environments affected by complex shadow conditions. The proposed framework is designed to mitigate systematic underestimation of UTCC caused by building-cast shadows, especially dark shadows where spectral and visual information is severely degraded.Includes Google Earth imagery and annotated datasets, along with a configuration guide, sample data, and visualization scripts.
# Environment configuration
-python=3.8

-torch==2.0.1+cu117

-torchvision==0.15.2+cu117

-torchaudio==2.0.2+cu117

-numpy=1.24

-pandas=2.0

-scipy=1.9

-scikit-learn=1.3

-matplotlib=3.7

-seaborn=0.13

-gdal=3.6

-rasterio=1.3

-geopandas=0.13

# weight 
Download our pretrained model:Canopy segmentation.pth is the weight for tree canopy extraction, while resnet101.pth is the weight of the pre-trained model obtained from the official website.
https://drive.google.com/drive/folders/1gb-YmrIt7odIPISsmyJE7rlOXu9P72iL?usp=sharing
# Our datasets (Download from Google drive)
https://drive.google.com/drive/folders/1e05a5CW8ETXhn6iThyvaiD9CxkbNK3xK?usp=sharing
# Our results
![Figure 15 ](https://github.com/user-attachments/assets/35089e53-f15e-4ab6-a00d-01f4f6c1eb33)
# Acknowledgement
This research  based on the code of  :https://github.com/nkszjx/UTB_master
