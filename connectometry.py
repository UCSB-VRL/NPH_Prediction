import nibabel as nib
import numpy as np
import os
from subprocess import call


def get_ventricles(BASE):
	'''
	Collects ventricles after segmentation, for connectomic analysis.
	'''
	path = os.path.join(BASE, 'transformed_outputs')
	outpath = os.path.join(BASE,'transformed_outputs','ventricles')

	if not os.path.exists(outpath):
		os.mkdir(outpath)

	for f in os.listdir(path):
		new_path = os.path.join(outpath, f)
		if not f.endswith('.nii.gz') or os.path.exists(new_path):
			continue
		fpath = os.path.join(path, f)
		img = nib.load(fpath)
		imarray = np.asanyarray(img.dataobj)
		imarray[np.where(imarray!=1)] = 0
		new_img = nib.Nifti1Image(imarray, img.affine)
		nib.save(new_img, new_path)



def run_connectometry(BASE):
	'''
	Outputs connectivity metrics of each patient.
	'''
	segpath = os.path.join(BASE, 'transformed_outputs', 'ventricles')
	dsistudio_path = os.path.join(BASE, 'dsistudio', 'build', 'dsi_studio')
	fibpath = os.path.join(BASE, 'template.fib.gz.mean.fib.gz')
	outpath = os.path.join(BASE, 'connectivity_metrics')

	if not os.path.exists(outpath):
		os.mkdir(outpath)

	for f in os.listdir(segpath):
		print(f)
		if not f.endswith('.nii.gz'):
			continue
		fpath = os.path.join(segpath, f)
		call([dsistudio_path, '--action=trk', '--source='+fibpath, '--seed_count=1000000', '--roa='+fpath, '--output=no_file', '--connectivity=AAL2'])
	call(['mv', '*.network_measures.txt', outpath]) 
