## White Matter, Ventricle and Subarachnoid Space Segmentation of CT scans, with Normal Pressure Hydrocephalus Predicton


Code implementing the algorithm described in the paper 

*Zhang et. al., Fully Automated Volumetric Classification in CT Scans for Diagnosis and Analysis of Normal Pressure Hydrocephalus. https://arxiv.org/abs/1901.09088*


### Requirements


To download all prerequisites, in the terminal type
```
pip install -r requirements.txt
```
FSL is needed to run this code. Please go to the FSL website at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki to download their software.

Note that there is a current known bug to using FSL which requires the user to manually install libopenblas.

### Usage


All of the CT scans must be in compressed nifti (.nii.gz) format, in a folder named 'Scans'.
Please note the directory containing the 'Scans' folder and use that as an argument `--directory=</path/to/directory>` when running the code.
The main function nph_prediction.py takes three arguments from the command line: 

directory `--directory=</path/to/directory>`, 

model `--model=<model>`, where `<model>` can be `rf`, `linear_svm`, or `rbf_svm`, and 

`--parallel`, which is `True` if typed and `False` otherwise.

To run with default settings (recommended), type
```
python nph_prediction.py
```

For help on the settings, type
```
python nph_prediction.py -h
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
Po-yu Kao, CT registration

Austin Mcever, testing

##### Acknowledgements
Thanks to the Vision Research Laboratory at the University of California, Santa Barbara; Dr. Ashu Shelat at the Cottage Hospital of Santa Barbara; and Dr. Jeff Chen at the University of California, Irvine for their help and support.
