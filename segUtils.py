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

import numpy as np
import nibabel as nib
import pickle
import os
import warnings
from skimage.draw import ellipse
from cv2 import fastNlMeansDenoising as denoising
from scipy.ndimage import binary_dilation as dilation
from scipy.ndimage import binary_erosion as erosion
from scipy.ndimage import binary_opening as opening
from scipy.ndimage import binary_closing as closing
from sklearn.ensemble import RandomForestClassifier
from skimage.draw import ellipse
from skimage.filters import gaussian, median
from skimage.morphology import disk, convex_hull_image
from skimage.segmentation import morphological_chan_vese as mcv
from joblib import Parallel, delayed
from scipy.ndimage import binary_fill_holes as fill_holes


def threshold(BASE, folder, parallel):
	'''
	Classification of tissue types in CT scan.
	'''
	classifier_name = os.path.join(BASE, 'TissueClassifier.pkl')
	if not os.path.exists(os.path.join(BASE, 'Thresholds')):
		os.mkdir(os.path.join(BASE, 'Thresholds'))
	#load tissue classifier
	with open(classifier_name, 'rb') as f:
		clf = pickle.load(f)
	fpath = os.path.join(BASE,folder)
	imnames = [os.path.join(BASE, 'Scans', f) for f in os.listdir(fpath) if (f.endswith('.nii.gz') or f.endswith('.nii'))]
	imnames.sort()
	with open(os.path.join(BASE, 'imname_list.pkl'), 'wb') as f:
		pickle.dump(imnames, f)

	# Apply Threshold
	print('-------- Applying Threshold --------')
	def apply_thresh(i):
		imname = imnames[i]
		imname_short = os.path.split(imname)[-1]
		print(imname_short)
		threshold_namev = os.path.join(BASE, 
									'Thresholds', 
									imname_short[:imname_short.find('.nii.gz')] +'.thresholdedv.nii.gz')
		threshold_namec = os.path.join(BASE,
									'Thresholds',
									imname_short[:imname_short.find('.nii.gz')] +'.thresholdedc.nii.gz')
		short_tnamev = os.path.split(threshold_namev)[-1]
		short_tnamec = os.path.split(threshold_namec)[-1]
		if short_tnamev in os.listdir(os.path.join(BASE, 'Thresholds')):
			if short_tnamec in os.listdir(os.path.join(BASE, 'Thresholds')):
				return
		if not os.path.exists(imname):
			print('does not exist')
			return
		im = nib.load(imname)
		image = im.get_data()
		image[np.where(image > 127)] = 127
		image[np.where(image < -128)] = -1000
		#denoising
		for s in range(0, image.shape[2]):
			slic = image[:,:,s]
			slic = np.uint8(slic)
			slic = denoising(slic, h=5)
			image[:,:,s] = np.float64(slic)
		#done denoising
		affine = im.affine
		header = im.header
		xsize, ysize, zsize = image.shape
		x = image.flatten()
		x_predict = x.reshape(-1,1)
		x_predict = x_predict.astype(float)
		y = clf.predict(x_predict)
		skull = np.copy(y)
		yv = np.copy(y)
		yc = np.copy(y)
		yv[np.where(yv != 1)[0]] = 0
		yc[np.where(yc != 2)[0]] = 0
		yc[np.where(yc == 2)[0]] = 1
		skull[np.where(skull != 3)[0]] = -1
		skull[np.where(skull == 3)[0]] = 1
		threshold_imgv = yv.reshape(image.shape)
		threshold_imgc = yc.reshape(image.shape)
		skull_img = skull.reshape(image.shape)
		structure = np.array([[1,1,1],[1,1,1],[1,1,1]])
		threshold_imgv[np.where(threshold_imgv < 0.5)] = -1
		threshold_imgv[np.where(threshold_imgc > 0.5)] = -1
		threshold_imgc[np.where(threshold_imgc < 0.5)] = -1
		threshold_imgc[np.where(threshold_imgv > 0.5)] = -1
		skull_name = os.path.join(BASE, 
								'Thresholds', 
								imname_short[:imname_short.find('.nii.gz')]+'.skull.nii.gz')
		nii_imagev = nib.Nifti1Image(threshold_imgv.astype(np.float32), affine, header)
		nii_imagec = nib.Nifti1Image(threshold_imgc.astype(np.float32), affine, header)
		skull_image = nib.Nifti1Image(skull_img.astype(np.float32), affine, header)
		nib.save(nii_imagev, threshold_namev)
		nib.save(nii_imagec, threshold_namec)
		nib.save(skull_image, skull_name)
		print('done thresholding: ' + imname)

	if parallel:
		Parallel(n_jobs=4)(delayed(apply_thresh)(i) for i in range(0, len(imnames)))
	else:
		for i in range(0, len(imnames)):
			apply_thresh(i)


def subarachnoid_seg(BASE, seg_model, parallel):
	'''
	Segments the subarachnoid space after white matter and ventricle segmentation.
	'''
	print('---------------- Subarachnoid Segmentation ------------------')
	imnames = pickle.load(open(os.path.join(BASE,'imname_list.pkl'), 'rb'))
	imnames.sort()

	def subseg(i):
		imname = imnames[i]
		imname_short = os.path.split(imname)[-1]
		print(imname_short)
		if seg_model == 'mcv':
			threshold_name = os.path.join(BASE,
									'Thresholds',
									imname_short[:imname_short.find('.nii.gz')] + '.skull.nii.gz')
			new_name = threshold_name[:threshold_name.find('.skull.nii.gz')] + '.skull1.nii.gz'
		else:
			threshold_name = os.path.join(BASE, 'UNet_Outputs', imname_short[:imname_short.find('.nii.gz')] + '.segmented.nii.gz')
			new_name = os.path.join(BASE, 'Thresholds', imname_short[:imname_short.find('.nii.gz')] + '.brain.nii.gz')
		if not os.path.exists(threshold_name):
			print('skipped due to no threshold')
			return
		threshold_image = nib.load(threshold_name)
		threshold_array = threshold_image.get_data()
		threshold_namev = os.path.join(BASE,
									'Thresholds',
									imname_short[:imname_short.find('.nii.gz')] + '.thresholdedv.nii.gz')
		threshold_imagev = nib.load(threshold_namev)
		varray = threshold_imagev.get_data()
		if seg_model == 'unet':
			final_pred = 'UNet_Outputs'
		else:
			final_pred = 'Final_Predictions'
		segment_name = os.path.join(BASE,
									final_pred,
									imname_short[:imname_short.find('.nii.gz')] + '.segmented.nii.gz')
		#orig_vname = os.path.join(BASE,
		#							'Predictions',
		#							imname_short[:imname_short.find('.nii.gz')] + '.segmentedv150.nii.gz')
		new_segname = segment_name[:segment_name.find('.nii.gz')] + '1.nii.gz'
		if not os.path.exists(segment_name):
			print('skipped due to no segment')
			return
		if os.path.exists(new_segname):
			return
		segment_img = nib.load(segment_name)
		segment_array = segment_img.get_data()
		#orig_vimg = nib.load(orig_vname)
		#orig_varray = orig_vimg.get_data()
		thresh_filled = np.copy(threshold_array)
		thresh_filled[np.where(threshold_array<0)] = 0
		thresh_filled[np.where(segment_array>0)] = 1
		if seg_model == 'unet':
			c_matter_z = np.where(segment_array==2)[2]
			if c_matter_z.size == 0:
				print('skipping due to no vent in segment')
				return
			r = range(c_matter_z.min(), c_matter_z.max())
		else:
			r = range(65, 182)
		for s in r:
			slic = thresh_filled[:,:,s]
			slic = fill_holes(slic)
			thresh_filled[:,:,s] = slic
		for s in range(0, thresh_filled.shape[2]):
			slic = thresh_filled[:,:,s]
			if seg_model == 'unet':
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					slic = convex_hull_image(slic)
			else:
				segslic = segment_array[:,:,s]
				seg_inds = np.where(segslic > 0.5)
				if len(seg_inds[0]) < 1:
					thresh_filled[:,:,s] = 0
					continue
				x_min = np.min(seg_inds[0])
				x_max = np.max(seg_inds[0])
				y_min = np.min(seg_inds[1])
				y_max = np.max(seg_inds[1])
				slic[0:x_min,:] = 0
				slic[:,0:y_min] = 0
			thresh_filled[:,:,s] = slic
		subarray = np.copy(varray)
		subarray[np.where(segment_array > 0)] = 0
		subarray[np.where(thresh_filled < 0.5)] = 0
		varray[np.where(segment_array > 0)] = -1
		varray[np.where(thresh_filled < 0.5)] = -1

		new_thresholdv = nib.Nifti1Image(varray, threshold_imagev.affine, threshold_imagev.header)
		new_tnamev = os.path.join(BASE,
								'Thresholds',
								imname_short[:imname_short.find('.nii.gz')] + '.thresholdedv1.nii.gz')
		nib.save(new_thresholdv, new_tnamev)

		segment_array[np.where(subarray > 0.5)] = 3
		segment_array[np.where((varray==1) & (segment_array==1))] = 3

		if seg_model == 'mcv':
			segment_array[:,:,0:40] = 0

		segment_img = nib.Nifti1Image(segment_array, segment_img.affine, segment_img.header)
		filled_image = nib.Nifti1Image(thresh_filled, threshold_image.affine, threshold_image.header)
		nib.save(filled_image, new_name)
		nib.save(segment_img, new_segname)

	r = range(0, len(imnames))
	if parallel:
		Parallel(n_jobs=5)(delayed(subseg)(i) for i in r)
	else:
		for k in r:
			subseg(k)


def combine_thresh(BASE):
	'''
	Combines White Matter and CSF threshold masks.
	'''
	imnames = pickle.load(open('imname_list1.pkl', 'rb'), encoding='latin1')
	imnames.sort()
	segfiles = os.listdir(os.path.join(BASE, 'Thresholds'))

	count = 0
	for imname in imnames:
		if 'NAV' in imname:
			continue
		imname_short = os.path.split(imname)[-1]
		print(str(count) + ': ' + imname_short)
		count += 1
		vsegname = os.path.join(BASE, 
					'Thresholds',
					imname_short[:imname_short.find('_MNI152.nii.gz')] + '.thresholdedv.nii.gz')
		csegname = os.path.join(BASE, 
					'Thresholds',
					imname_short[:imname_short.find('_MNI152.nii.gz')] + '.thresholdedc.nii.gz')
		if os.path.split(vsegname)[-1] not in segfiles or os.path.split(csegname)[-1] not in segfiles:
			print('skipped')
			continue
		vsegimg = nib.load(vsegname)
		vsegarr = vsegimg.get_data()
		vsegheader = vsegimg.header
		vsegaffine = vsegimg.affine
		vsegarr[np.where(vsegarr != 1)] = 0

		csegimg = nib.load(csegname)
		csegarr = csegimg.get_data()
		csegarr[np.where(csegarr != 1)] = 0
		vsegarr[np.where(csegarr > 0)] = 0

		segarr = vsegarr + 2*csegarr

		segname = os.path.join(BASE, 
					'Combined_Thresholds',
					imname_short[:imname_short.find('.nii.gz')] + '.segmented.nii.gz')

		segimg = nib.Nifti1Image(segarr, vsegaffine, vsegheader)
		nib.save(segimg, segname)


def combine_segs(BASE):
	'''
	Combines white matter and ventricle segmentations.
	'''
	if not os.path.exists(os.path.join(BASE, 'Final_Predictions')):
		os.mkdir(os.path.join(BASE, 'Final_Predictions'))
	imnames = pickle.load(open(os.path.join(BASE,'imname_list1.pkl'), 'rb'))
	imnames.sort()
	segfiles = os.listdir(os.path.join(BASE, 'Predictions'))

	count = 0
	for imname in imnames:
		if 'NAV' in imname:
			continue
		imname_short = os.path.split(imname)[-1]
		print(str(count) + ': ' + imname_short)
		count += 1
		vsegname = os.path.join(BASE, 
					'Predictions',
					imname_short[:imname_short.find('.nii.gz')] + '.segmentedv150_1.nii.gz')
		csegname = os.path.join(BASE, 
					'Predictions',
					imname_short[:imname_short.find('.nii.gz')] + '.segmentedc150.nii.gz')
		if os.path.split(vsegname)[-1] not in segfiles or os.path.split(csegname)[-1] not in segfiles:
			print('skipped')
			continue
		vsegimg = nib.load(vsegname)
		vsegarr = vsegimg.get_data()
		vsegheader = vsegimg.header
		vsegaffine = vsegimg.affine

		csegimg = nib.load(csegname)
		csegarr = csegimg.get_data()
		vsegarr[np.where(csegarr > 0)] = 0

		segarr = vsegarr + 2*csegarr

		segname = os.path.join(BASE, 
					'Final_Predictions',
					imname_short[:imname_short.find('.nii.gz')] + '.segmented.nii.gz')
		segimg = nib.Nifti1Image(segarr, vsegaffine, vsegheader)
		nib.save(segimg, segname)


def modify_image(seg_name, imname, segclass):
	'''
	Modifies the segmentation in the case of previous stroke.
	'''
	seg_img = nib.load(seg_name)
	segarray = seg_img.get_data()
	'''
	if segclass == 'v':
		xsize, ysize, zsize = segarray.shape
		if np.sum(segarray) > 90000:
			orig_img = nib.load(imname).get_data()
			#denoising
			orig_img[np.where(orig_img > 127)] = 127
			orig_img[np.where(orig_img < -128)] = -128
			orig_img = orig_img + 128
			orig_img = np.uint8(orig_img)
			for ind in range(0, orig_img.shape[2]):	
				slic = orig_img[:,:,ind]
				slic = denoising(slic, h=5)
				orig_img[:,:,ind] = slic
			orig_img = np.float32(orig_img)
			orig_img = orig_img - 128
			orig_img[np.where(orig_img == -128)] = -1000
			#done denoising
			segarray[np.where(orig_img>10)] = 0
			segarray[:,:,120:int(zsize)-1] = 0
			segarray[:,:,0:65] = 0
			#dilation
			for z in range(zsize):
				slic = segarray[:,:,z]
				slic = dilation(slic, iterations=5)
				segarray[:,:,z] == slic
	'''
	new_segimg = nib.Nifti1Image(segarray, seg_img.affine, seg_img.header)
	new_segname = seg_name[:seg_name.find('.nii.gz')]+'_1.nii.gz'
	nib.save(new_segimg, new_segname)


def snake_seg(BASE, PARALLEL=True, segclass='v'):
	'''
	Morphological segmentation based on a-priori seeding of white matter and ventricles in CT Scans.
	'''
	classifier_name = 'TissueClassifier.pkl'
	if not os.path.exists(os.path.join(BASE, 'Thresholds')):
		os.mkdir(os.path.join(BASE, 'Thresholds'))
	if not os.path.exists(os.path.join(BASE, 'Predictions')):
		os.mkdir(os.path.join(BASE, 'Predictions'))
	#load tissue classifier
	with open(classifier_name, 'rb') as f:
		clf = pickle.load(f)

	affine_dict = pickle.load(open(os.path.join(BASE, 'imname_affine.pkl'), 'rb'))
	header_dict = pickle.load(open(os.path.join(BASE, 'imname_header.pkl'), 'rb'))
	imnames = pickle.load(open(os.path.join(BASE, 'imname_list.pkl'), 'rb'))
	imnames.sort()

	# Apply Threshold
	print('-------- Applying Threshold --------')
	def apply_thresh(i):
		imname = imnames[i]
		imname_short = os.path.split(imname)[-1]
		print(imname_short)
		threshold_namev = os.path.join(BASE, 
									'Thresholds', 
									imname_short[:imname_short.find('.nii.gz')]+'.thresholdedv.nii.gz')
		threshold_namec = os.path.join(BASE,
									'Thresholds',
									imname_short[:imname_short.find('.nii.gz')]+'.thresholdedc.nii.gz')
		short_tnamev = os.path.split(threshold_namev)[-1]
		short_tnamec = os.path.split(threshold_namec)[-1]
		if short_tnamev in os.listdir(os.path.join(BASE, 'Thresholds')):
			if short_tnamec in os.listdir(os.path.join(BASE, 'Thresholds')):
				return
		if not os.path.exists(imname):
			print('does not exist')
			return
		image = nib.load(imname).get_data()
		image[np.where(image > 127)] = 127
		image[np.where(image < -128)] = -1000
		affine = affine_dict[imname]
		header = header_dict[imname]
		xsize, ysize, zsize = image.shape
		x = image.flatten()
		x_predict = x.reshape(-1,1)
		x_predict = x_predict.astype(float)
		y = clf.predict(x_predict)
		skull = np.copy(y)
		yv = np.copy(y)
		yc = np.copy(y)
		yv[np.where(yv != 1)[0]] = 0
		yc[np.where(yc != 2)[0]] = 0
		yc[np.where(yc == 2)[0]] = 1
		skull[np.where(skull != 3)[0]] = -1
		skull[np.where(skull == 3)[0]] = 1
		threshold_imgv = yv.reshape(image.shape)
		threshold_imgc = yc.reshape(image.shape)
		skull_img = skull.reshape(image.shape)
		structure = np.array([[1,1,1],[1,1,1],[1,1,1]])
		threshold_imgv[np.where(threshold_imgv < 0.5)] = -1
		threshold_imgv[np.where(threshold_imgc > 0.5)] = -1
		threshold_imgc[np.where(threshold_imgc < 0.5)] = -1
		threshold_imgc[np.where(threshold_imgv > 0.5)] = -1
		skull_name = os.path.join(BASE, 
								'Thresholds', 
								imname_short[:imname_short.find('.nii.gz')]+'.skull.nii.gz')
		nii_imagev = nib.Nifti1Image(threshold_imgv.astype(np.float32), affine, header)
		nii_imagec = nib.Nifti1Image(threshold_imgc.astype(np.float32), affine, header)
		skull_image = nib.Nifti1Image(skull_img.astype(np.float32), affine, header)
		nib.save(nii_imagev, threshold_namev)
		nib.save(nii_imagec, threshold_namec)
		nib.save(skull_image, skull_name)
		print('done thresholding: ' + imname)

	if PARALLEL:
		Parallel(n_jobs=4)(delayed(apply_thresh)(i) for i in range(0, len(imnames)))
	else:
		for i in range(0, len(imnames)):
			apply_thresh(i)

	with open(os.path.join(BASE, 'imname_affine1.pkl'), 'wb') as f:
		pickle.dump(affine_dict, f)
	with open(os.path.join(BASE, 'imname_header1.pkl'), 'wb') as f:
		pickle.dump(header_dict, f)
	with open(os.path.join(BASE, 'imname_list1.pkl'), 'wb') as f:
		pickle.dump(imnames, f)

	# Active Contour
	print('--------- Snake Seg' + segclass + ' ---------')
	def store_evolution_in(lst):
		def _store(x):
			lst.append(np.copy(x))
		return _store


	def seg_ventricle(i):
		evolution = []
		callback = store_evolution_in(evolution)
		imname = imnames[i]

		imname1 = os.path.split(imname)[-1]
		threshold_name = os.path.join(BASE, 
									'Thresholds', 
									imname1[:imname1.find('.nii.gz')]+'.thresholded' + segclass + '.nii.gz')
		seg_name = os.path.join(BASE, 
								'Predictions', 
								imname1[:imname1.find('.nii.gz')]+'.segmented' + segclass + '0.nii.gz')
		if os.path.exists(seg_name):
			return
		print('starting segmentation: ' + imname)

		timg = nib.load(threshold_name).get_data()
	
		#img = nib.load(imname).get_data()
		#timg[np.where(timg>0)] = img[np.where(timg>0)]
		mask_name = os.path.join(BASE, 'Anatomical_Mask.nii.gz')
		anatomical_mask = nib.load(mask_name).get_data()
		affine = affine_dict[imname]
		header = header_dict[imname]
		xsize, ysize, zsize = timg.shape
		initial_ls = np.zeros(timg.shape)
		radius = int(xsize*0.05)
		radx = max(2, int(radius/abs(affine[0,0])))
		rady = max(2, int(radius/abs(affine[1,1])))
		radz = max(2, int(radius/abs(affine[2,2])))
		# Drawing 3-D ellipses as seeds
		if segclass == 'v':
			timg[85:100, 100:218, 0:65] = -1
			timg[90:94, 0:80, 60:90] = -1
			timg[75:85, 110:218, 0:65] = -1
			timg[100:110, 110:218, 0:65] = -1
			timg[7:110, 60:97, 0:65] = -1
			timg[np.where(anatomical_mask > 0.5)] = -1
			index = (int(xsize*.5), int(ysize*.5), int(zsize*.5))
			rr1, cc1 = ellipse(index[0], index[1], radx, rady)
			rr2, cc2 = ellipse(index[0], index[2], radx, radz)
			rr3, cc3 = ellipse(int(index[0]-radx), int(index[1]-rady), radx, rady)
			rr4, cc4 = ellipse(int(index[0]-radx), index[2], radx, radz)
			rr5, cc5 = ellipse(int(index[0]-radx), int(index[1]+rady), radx, rady)
			rr6, cc6 = ellipse(int(index[0]+radx), int(index[1]-rady), radx, rady)
			rr7, cc7 = ellipse(int(index[0]+radx), index[2], radx, radz)
			rr8, cc8 = ellipse(int(index[0]+radx), int(index[1]+rady), radx, rady)
			rr9, cc9 = ellipse(91, 85,radx, rady)
			rr10, cc10 = ellipse(91, 75, radx, radz)

			temp = np.zeros(initial_ls.shape)
			temp2 = np.zeros(temp.shape)
			initial_ls[rr1, cc1, :] = 1
			temp[rr2, :, cc2] = 1
			initial_ls = np.multiply(initial_ls, temp)
			temp[:,:,:] = 0
			temp[rr3, cc3, :] = 1
			temp2[rr4, :, cc4] = 1
			initial_ls += np.multiply(temp, temp2)
			temp[:,:,:] = 0
			temp[rr5, cc5, :] = 1
			initial_ls += np.multiply(temp, temp2)
			temp[:,:,:] = 0
			temp2[:,:,:] = 0
			temp[rr6, cc6, :] = 1
			temp2[rr7, :, cc7] = 1
			initial_ls += np.multiply(temp, temp2)
			temp[:,:,:] = 0
			temp[rr8, cc8, :] = 1
			initial_ls += np.multiply(temp, temp2)

			temp[:,:,:] = 0
			temp2[:,:,:] = 0
			temp[rr9, cc9, :] = 1
			temp2[rr10, :, cc10] = 1
			initial_ls += np.multiply(temp, temp2)

		else:
			index = (int(xsize*.5), int(ysize*.75), int(zsize*.5))
			index1 = (int(xsize*.5), int(ysize*.5), int(zsize*.75))
			index2 = (int(xsize*.5), int(ysize*.25), int(zsize*.5))
			index3 = (int(xsize*.5), int(ysize*.75), int(zsize*.4))
			rr, cc = ellipse(index[0], index[1], radx, rady)
			rr1, cc1 = ellipse(index[0], index[2], radx, radz)
			rr2, cc2 = ellipse(index1[0], index1[1], radx, rady)
			rr3, cc3 = ellipse(index1[0], index1[2], radx, radz)
			rr4, cc4 = ellipse(index2[0], index2[1], radx, rady)
			rr5, cc5 = ellipse(index2[0], index2[2], radx, radz)
			rr6, cc6 = ellipse(index3[0], index3[1], radx, rady)
			rr7, cc7 = ellipse(index3[0], index3[2], radx, radz)
			temp = np.zeros(initial_ls.shape)
			temp1 = np.copy(temp)
			initial_ls[rr, cc, :] = 1
			temp[rr1, :, cc1] = 1
			temp1[rr3, :, cc3] = 1
			initial_ls = np.multiply(initial_ls, temp)
			temp = np.zeros(temp.shape)
			temp[rr2, cc2, :] = 1
			initial_ls += np.multiply(temp, temp1)
			temp = np.zeros(temp.shape)
			temp[rr4, cc4, :] = 1
			temp1 = np.zeros(temp1.shape)
			temp1[rr5, :, cc5] = 1
			initial_ls += np.multiply(temp, temp1)
			temp = np.zeros(temp.shape)
			temp1 = np.zeros(temp1.shape)
			temp[rr6, cc6, :] = 1
			temp1[rr7, :, cc7] = 1
			initial_ls += np.multiply(temp, temp1)

		initial_ls[np.where(timg <= 0)] = 0
		if segclass == 'v':
			numiter = 150
		else:
			numiter = 150
		segmentation = mcv(timg.astype(float), 
						iterations=numiter, 
						init_level_set=initial_ls,
						iter_callback=callback)

		segmentation[np.where(timg<=0)] = 0
		for i in range(0, len(evolution), len(evolution)-1):
			seg_img = evolution[i]
			seg_img[np.where(timg<=0)] = 0
			if segclass == 'v':
				seg_img[:,:,0:int(zsize*0.2)] = 0
			nii_seg = nib.Nifti1Image(seg_img, affine, header)
			seg_name = os.path.join(BASE, 
								'Predictions', 
								imname1[:imname1.find('.nii.gz')]+'.segmented' + segclass + str(i) + '.nii.gz')
			nib.save(nii_seg, seg_name)
		seg_name = os.path.join(BASE, 
								'Predictions', 
								imname1[:imname1.find('.nii.gz')] + '.segmented' + segclass + str(len(evolution)-1) + '.nii.gz')
		modify_image(seg_name, imname, segclass)	

		print('completed segmentation: ' + imname1)


	if PARALLEL:
		Parallel(n_jobs=4)(delayed(seg_ventricle)(i) for i in range(0, len(imnames)))
	else:
		for i in range(0, len(imnames)):
			seg_ventricle(i)
