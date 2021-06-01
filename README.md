## White Matter, Ventricle and Subarachnoid Space Segmentation of CT scans, with Normal Pressure Hydrocephalus Predicton


Code implementing the algorithm described in the paper 

*Zhang et. al., Automated Segmentation of Computed Tomography Scans and Connectivity Analysis for Normal Pressure Hydrocephalus*


### Requirements


To download all prerequisites, in the terminal type
```
pip install -r requirements.txt
```
In order to use either heatmaps or the NPH prediction portion of the code, FSL is needed to run this code. Please go to the FSL website at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki to download their software.

Note that there is a current known bug to using FSL which requires the user to manually install libopenblas.

To complete the NPH prediction portion of the module, DSI Studio is needed. To download or build DSI Studio, please go to http://dsi-studio.labsolver.org/dsi-studio-download and follow the relevant instructions for your operating system.

The unet model will not require FSL. However, it requires a large file, `unet_ce_hard_per_im_s8841_all/model_last.tar`, which can be downloaded manually on the github website or using `git-lfs`. 

The code has been tested only on python version 3.6.

### Usage


All of the CT scans must be in compressed nifti (.nii.gz) format, in a folder named 'Scans'.
Please note the directory containing the 'Scans' folder and use that as an argument `--directory=</path/to/directory>` when running the code.
The main function nph_prediction.py takes the following arguments from the command line: 

directory `--directory=</path/to/directory>`, 

seg_model `--seg_model=<model>`, where `<model>` can be `unet` or `mcv`,

--predict_nph, which will predict NPH if selected,

--use_heatmap, which will use heatmap to aid in segmentation when selected,

--dsi_dir, which is the path to your dsi studio folder which contains the build folder,

and

`--gpu`, which is `True` if typed and `False` otherwise.

To run with default settings (recommended), type
```
python3 nph_prediction.py
```

For help on the settings, type
```
python3 nph_prediction.py -h
```

The runtime is approximately 5 minutes per image with GPU.

The output will be a `.csv` file to `<folder>` with the name of the nifti image file, the white matter, ventricle, and subarachnoid space volumes, and another `.csv` file with the recommendation of 'normal' or 'possible NPH'.

An example CT scan has been included in the Scans directory. 

This is an open source image from http://headctstudy.qure.ai/dataset, accompanying paper at:

*Sasank Chilamkurthy et al. Deep learning algorithms for detection of critical findings in
head CT scans: a retrospective study. DOI:https://doi.org/10.1016/S0140-6736(18)31645-3.*

##### Authors
Angela Zhang

##### Contributors

Amil Khan, coding and illustrations

##### Acknowledgements
Thanks to the Vision Research Laboratory at the University of California, Santa Barbara; Dr. Ashu Shelat at the Cottage Hospital of Santa Barbara; and Dr. Jeff Chen at the University of California, Irvine for their help and support.

