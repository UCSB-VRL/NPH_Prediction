## White Matter, Ventricle and Subarachnoid Space Segmentation of CT scans, with Normal Pressure Hydrocephalus Prediction


Code implementing the algorithm described in the paper 

*Zhang et. al., Fully Automated Volumetric Classification in CT Scans for Diagnosis and Analysis of Normal Pressure Hydrocephalus. https://arxiv.org/abs/1901.09088*


### Requirements


To download all prerequisites, in the terminal type
```
pip install -r requirements.txt
```
In order to use the morphological chan-vese model, FSL is needed to run this code. Please go to the FSL website at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki to download their software.

Note that there is a current known bug to using FSL which requires the user to manually install libopenblas.

The unet model will not require FSL. However, it requires a large file, `unet_ce_hard_per_im_s8841_all/model_last.tar`, which can be downloaded manually on the github website or using `git-lfs`. 

The code has been tested only on python version 3.6.

### Usage


All of the CT scans must be in compressed nifti (.nii.gz) format, in a folder named 'Scans'.
Please note the directory containing the 'Scans' folder and use that as an argument `--directory=</path/to/directory>` when running the code.
The main function nph_prediction.py takes the following arguments from the command line: 

directory `--directory=</path/to/directory>`, 

seg_model `--seg_model=<model>`, where `<model>` can be `unet` or `mcv`,

`--parallel`, which is `True` if typed and `False` otherwise, and

`--gpu`, which is `True` if typed and `False` otherwise.

To run with default settings (recommended), type
```
python3 nph_prediction.py
```

For help on the settings, type
```
python3 nph_prediction.py -h
```

The runtime is approximately 10 minutes per image.

The output will be a `.csv` file to `<folder>` with the name of the nifti image file, the white matter, ventricle, and subarachnoid space volumes, and another `.csv` file with the recommendation of 'normal' or 'possible NPH'.

An example CT scan has been included in the Scans directory. 

This is an open source image from http://headctstudy.qure.ai/dataset, accompanying paper at:

*Sasank Chilamkurthy et al. Deep learning algorithms for detection of critical findings in
head CT scans: a retrospective study. DOI:https://doi.org/10.1016/S0140-6736(18)31645-3.*

##### Authors
Angela Zhang

##### Contributors
Po-yu Kao, CT registration, UNet pretraining

Austin Mcever, testing

##### Acknowledgements
Thanks to the Vision Research Laboratory at the University of California, Santa Barbara; Dr. Ashu Shelat at the Cottage Hospital of Santa Barbara; and Dr. Jeff Chen at the University of California, Irvine for their help and support.

