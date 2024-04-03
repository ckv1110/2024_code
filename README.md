# Repository for the integration of various models in the PhD project.
## Collection of code that I had and will use for the integration of various models into the final pipeline.
### What is in this repository:
- Various U-Net models built on [Pytorch](https://pytorch.org/docs/stable/index.html)
- Default pre-trained [StarDist](https://github.com/stardist/stardist) model for nuclei segmentation
- Stain Normatlizaiton + Deconvolution + Nuclei feature extractor based on [HistomicsTK](https://digitalslidearchive.github.io/HistomicsTK/index.html)
- Code snippet to visualize models and check them, saved as .onnx files
### What works:
- Base U-Net model fully working
- StarDist whole WSI nuclei segmentation + Stain normalization + nuclei feature extraction (60 features in total) pipeline is working
### Currently working on:
- Building more models into __model_utils.py__ from scratch => Gives more granular control over what to tune 
    - Build ResU-Net and various ResNet (18, 34, 50, 101) encoder U-Net's
    - Build fusion blocks for multi-task learning (example from https://doi.org/10.1016/j.media.2022.102481)
    ![https://doi.org/10.1016/j.media.2022.102481](https://ars.els-cdn.com/content/image/1-s2.0-S1361841522001281-gr1_lrg.jpg)
        - This will allow me to use the H&E and nuclei images to train 2 models simultaneously to predict TMEs and marker heatmaps whilst having the 2 models share information. Thus, theoretically resulting in a more robust model. 
    - Implement [FOVeation](https://github.com/lxasqjc/Foveation-Segmentation) module into the model  
    - Build training script using [Pytorch Lightning](https://lightning.ai/pytorch-lightning)
- Implement nuclei centroid for the __feature_extractor.py__ utility file
    - Create contours around tumour segmentation mask and ray cast nuclei centroid coordinates to spatially map them
- Convert IHC stains into heatmaps for training + automate script
- Test out [Stain2Stain](https://github.com/pegahs1993/Stain-to-Stain-Translation) GAN stain normalization method (_Low Prio_ - normalization quality is already satisfactory apart from some minor nitpicks)
### What is required to run these codes:
#### Nuclei segmentation and feature extractor
- __Conda__ environment with __Python 3.9__ ready
- ```conda install -c conda-forge cudnn==8.4.1.50, cudatoolkit==11.7.0``` => If you want to use NVIDIA GPU processes, otherwise skip
- ```python -m pip install tensorflow==2.10.0```
- ```python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels```

#### Pytorch Models
- Seperate __Conda__ environment with __Python >3.9__ ready just in case it conflicts with Nuclei environment
- For NVIDIA GPU users:
    - ```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```
- Non-GPU:
    - ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```
- ```python -m pip install torch```
- ```python -m pip install lightning```
- ```python -m pip install netron```
- ```python -m pip install onnx```
- ```python -m pip install torchvision```
 

