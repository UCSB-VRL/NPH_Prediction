import nibabel as nib
import numpy as np
import os
from subprocess import call


def transform_seg(BASE):
	'''
	Transforms segmentations to standard space for connectometry analysis.
	'''
	print('-------- Transforming Segmentations --------')
	MNI_152 = os.path.join(BASE, 'MNI152_T1_1mm.nii.gz')
	segpath = os.path.join(BASE, 'UNet_Outputs')
	affinepath = os.path.join(BASE, 'MNI152')
	for seg in os.listdir(segpath):
		full_segpath = os.path.join(segpath, seg)
		affine_mtx = os.path.join(affinepath, seg[:seg.find('.segmented.nii.gz')]+'_affine.mat')
		transformed_segpath = os.path.join(BASE, 'transformed_outputs')
		if not os.path.exists(transformed_segpath):
			os.mkdir(transformed_segpath)
		if not os.path.exists(affine_mtx):
			print('affine matrix missing! ' + affine_mtx)
			continue
		print(seg)
		transformed_name = os.path.join(transformed_segpath, seg)
		call(['flirt', '-in', full_segpath, '-ref', MNI_152, '-applyxfm', '-init', affine_mtx, '-out', transformed_name])
		


def get_ventricles(BASE):
	'''
	Collects ventricles after segmentation, for connectomic analysis.
	'''
	print('---------- Collecting Ventricles ----------')
	path = os.path.join(BASE, 'transformed_outputs')
	outpath = os.path.join(BASE,'transformed_outputs','ventricles')

	if not os.path.exists(outpath):
		os.mkdir(outpath)

	for f in os.listdir(path):
		new_path = os.path.join(outpath, f)
		if not f.endswith('.nii.gz') or os.path.exists(new_path):
			continue
		print(f)
		fpath = os.path.join(path, f)
		img = nib.load(fpath)
		imarray = np.asanyarray(img.dataobj)
		imarray[np.where(imarray!=1)] = 0
		new_img = nib.Nifti1Image(imarray, img.affine)
		nib.save(new_img, new_path)



def run_connectometry(BASE, dsi_dir):
	'''
	Outputs connectivity metrics of each patient.
	'''
	print('--------- running connectometry ----------')
	segpath = os.path.join(BASE, 'transformed_outputs', 'ventricles')
	dsistudio_path = os.path.join(dsi_dir, 'build', 'dsi_studio')
	fibpath = os.path.join(BASE, 'template.fib.gz.mean.fib.gz')
	outpath = os.path.join(BASE, 'connectivity_metrics')

	if not os.path.exists(outpath):
		os.mkdir(outpath)

	for f in os.listdir(segpath):
		if not f.endswith('.nii.gz'):
			continue
		fpath = os.path.join(segpath, f)
		print(fpath)
		#call(['singularity', 'exec', 'dsistudio_latest.sif', '--action=trk', '--source='+fibpath, '--seed_count=1000000', '--roa='+fpath, '--output=no_file', '--connectivity=AAL2', '>', os.path.join(outpath, 'network_measures.txt')])
		call([dsistudio_path, '--action=trk', '--source='+fibpath, '--seed_count=1000000', '--roa='+fpath, '--output=no_file', '--connectivity=AAL2'])
	move_files = [f for f in os.listdir(os.getcwd()) if f.endswith('network_measures.txt') or f.endswith('connectogram.mat')]
	new_path = os.path.join(outpath, f+'network_measures.txt')
	for f in move_files:
		if f.endswith('connectogram.mat'):
			call(['rm', f])
		else:
			call(['mv', f, new_path]) 
