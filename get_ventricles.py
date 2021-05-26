import nibabel as nib
import numpy as np
import os
import ipdb


def get_ventricles(BASE):
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

	print('done')
