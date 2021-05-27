###############################################################################
##  Vision Research Laboratory and                                           ##
##  Center for Multimodal Big Data Science and Healthcare                    ##
##  University of California at Santa Barbara                                ##
## ------------------------------------------------------------------------- ##
##                                                                           ##
##     Copyright (c) 2019                                                    ##
##     by the Regents of the University of California                        ##
##                            All rights reserved                            ##
##                                                                           ##
## Redistribution and use in source and binary forms, with or without        ##
## modification, are permitted provided that the following conditions are    ##
## met:                                                                      ##
##                                                                           ##
##     1. Redistributions of source code must retain the above copyright     ##
##        notice, this list of conditions, and the following disclaimer.     ##
##                                                                           ##
##     2. Redistributions in binary form must reproduce the above copyright  ##
##        notice, this list of conditions, and the following disclaimer in   ##
##        the documentation and/or other materials provided with the         ##
##        distribution.                                                      ##
##                                                                           ##
##                                                                           ##
## THIS SOFTWARE IS PROVIDED BY <COPYRIGHT HOLDER> "AS IS" AND ANY           ##
## EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE         ##
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR        ##
## PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> OR           ##
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,     ##
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       ##
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        ##
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    ##
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      ##
## NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        ##
## SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              ##
##                                                                           ##
## The views and conclusions contained in the software and documentation     ##
## are those of the authors and should not be interpreted as representing    ##
## official policies, either expressed or implied, of <copyright holder>.    ##
###############################################################################

import subprocess
import nibabel as nib
import numpy as np
import pickle
import os


def affine_transform(BASE):
	'''
	calculates and applies an affine transform of the CT scans in 'BASE/Scans' to MNI152 space.
	'''
	print('--------- Calculating Affine Transforms ---------')
	numpy_images = {}
	affine_dict = {}
	header_dict = {}
	imnames = []
	scanpath = os.path.join(BASE, 'Scans')
	affinepath = os.path.join(BASE, 'MNI152')
	if not os.path.exists(affinepath):
		os.makedirs(affinepath)
	for scanname in os.listdir(scanpath):
		if 'skull' in scanname or 'MNI152' in scanname or not scanname.endswith('.nii.gz'):
			continue
		affinename = scanname[:scanname.find('.nii.gz')] + '_MNI152.nii.gz'
		scanimgpath = os.path.join(scanpath, scanname)
		new_imname = os.path.join(affinepath, affinename)
		scanimg = nib.load(scanimgpath)
		if np.unique(scanimg.get_data()).size < 5:
			print('skipping due to not enough values')
			continue
		imname = scanimgpath[:scanimgpath.find('.nii.gz')] + '_MNI152.nii.gz'
		if affinename not in os.listdir(affinepath):
			print(scanname)
			subprocess.call(['python', 'CT2MNI152Affine.py', scanimgpath])
			subprocess.call(['mv', imname, new_imname])
			# move affine matrix from Scans to MNI152 directory
			affine_matrix = scanimgpath[:scanimgpath.find('.nii.gz')] + '_affine.mat'
			new_affine_matrix = new_imname[:new_imname.find('_MNI152.nii.gz')] + '_affine.mat'
			subprocess.call(['mv', affine_matrix, new_affine_matrix])
			# remove skull image
			skull_imname = scanimgpath[:scanimgpath.find('.nii.gz')] + '_skull.nii'
			subprocess.call(['rm', skull_imname])
		try:
			image = nib.load(new_imname)
		except:
			print('transform didnt work')
			continue
		affine_dict[new_imname] = image.affine
		header_dict[new_imname] = image.header.copy()
		imnames.append(new_imname)
	with open(os.path.join(BASE, 'imname_affine.pkl'), 'wb') as f:
		pickle.dump(affine_dict, f)
	with open(os.path.join(BASE, 'imname_header.pkl'), 'wb') as f:
		pickle.dump(header_dict, f)
	with open(os.path.join(BASE, 'imname_list.pkl'), 'wb') as f:
		pickle.dump(imnames, f)


def reverse_transform(BASE):
	'''
	Transforms registered CT scans back to the original subject space.
	'''
	print('--------- Calculating Reverse Transforms ---------')
	scanpath = os.path.join(BASE, 'Scans')
	affinepath = os.path.join(BASE, 'MNI152')
	for scanname in os.listdir(scanpath):
		if scanname.endswith('.nii.gz') and 'MNI' not in scanname:
			newaffinepath = os.path.join(affinepath, scanname[:scanname.find('.nii.gz')] + '_affine.mat')
			if not os.path.exists(newaffinepath):
				continue
			print(scanname)
			invmtxpath = os.path.join(affinepath, scanname[:scanname.find('.nii.gz')] + '_inverse.mat')
			subprocess.call(['convert_xfm', '-omat', invmtxpath, '-inverse', newaffinepath])


def transform_heatmaps(BASE):
	'''
	Transforms heatmaps into original subject space.
	'''
	scanpath = os.path.join(BASE, 'Scans')
	heatpath = os.path.join(BASE, 'Heatmaps')
	affine_path = os.path.join(BASE, 'MNI152')
	transformed_heatpath = os.path.join(BASE, 'Transformed_Heatmaps')
	heatimgs = os.listdir(heatpath)

	if not os.path.exists(transformed_heatpath):
		os.mkdir(transformed_heatpath)

	for scanname in os.listdir(scanpath):
		if scanname.endswith('.nii.gz') and 'MNI' not in scanname:
			print(scanname)
			subjectpath = os.path.join(scanpath, scanname)
			affinename = scanname[:scanname.find('.nii.gz')] + '_MNI152.nii.gz'
			if affinename not in os.listdir(scanpath):
				continue
			nameOfInvMatrix = os.path.join(affine_path, scanname[:scanname.find('.nii.gz')] + '_inverse.mat')
			nameOfAffineMatrix = os.path.join(affine_path, scanname[:scanname.find('.nii.gz')] + '_affine.mat')
			call(['convert_xfm', '-omat', nameOfInvMatrix, '-inverse', nameOfAffineMatrix])
			for img in heatimgs:
				impath = os.path.join(heatpath, img)
				outpath = os.path.join(transformed_heatpath, img)
				subprocess.call(['flirt','-in', impath, '-ref', subjectpath, '-applyxfm', '-init', nameOfInvMatrix, '-out', outpath, '-interp', 'nearestneighbour'])
