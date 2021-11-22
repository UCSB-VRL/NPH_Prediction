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
import csv
import nibabel as nib
import os
import pandas
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as rf_classifier
from sklearn import preprocessing


def get_volumes(BASE):
	'''
	Obtains the volumes of the ventricle, subarachnoid space, and white matter given segmentations.
	Volumes are output in a csv file.
	'''
	print('------------ getting volumes ---------------')
	imnames = pickle.load(open(os.path.join(BASE,'imname_list.pkl'), 'rb'))
	imnames.sort()
	volume_csv = os.path.join(BASE, 'volumes.csv')
	csv_exists = os.path.exists(volume_csv)
	f = open(volume_csv, 'w')
	writer = csv.writer(f)
	writer.writerow(['Scan', 'Vent', 'Sub', 'White'])
	ventricle_volumes = []
	sub_volumes = []
	white_volumes = []

	for imname in imnames:
		imname_short = os.path.split(imname)[-1]
		print(imname_short)
		final_pred = 'UNet_Outputs'
		seg_name = os.path.join(BASE,
					final_pred,
					imname_short[:imname_short.find('.nii.gz')] + '.segmented1.nii.gz')

		if not os.path.exists(seg_name):
			print('skipping due to no segmentation')
			continue
		segimg = nib.load(seg_name)
		segarray = segimg.get_data()
		affine = segimg.affine
		vol_per_vox = np.abs(affine[0,0]*affine[1,1]*affine[2,2])
		ventricle = np.sum(segarray == 1)*vol_per_vox
		white_matter = np.sum(segarray == 2)*vol_per_vox
		subarachnoid = np.sum(segarray == 3)*vol_per_vox

		if white_matter <= 5e5:
			print('invalid scan due to no white matter.')
			continue
		elif ventricle <= 2:
			print('invalid scan due to no ventricle.')
			continue
		elif subarachnoid <= 2:
			print('invalid scan due to no subarachnoid.')
			continue
		elif ventricle > white_matter:
			print('possible issue due to ventricle being bigger than white matter.')
		else:
			ventricle_volumes.append(float(ventricle))
			sub_volumes.append(float(subarachnoid))
			white_volumes.append(float(white_matter))

		whole_brain = float(ventricle+subarachnoid+white_matter)
		writer.writerow([imname_short, str(ventricle), str(subarachnoid), str(white_matter)])
	f.close()


def make_prediction(BASE):
	'''
	Makes predictions of possible NPH/no NPH given the volume information obtained by get_volumes, output to predictions_$model$.csv.
	model options: linear_svm, rbf_svm, rf
	'''
	print('------------ making prediction -------------')
	#load classifier
	classifier_name = 'conn_model.pkl'
	vol_name = 'volumes_unet.csv'
	with open(os.path.join(BASE, 'nph_classifiers', classifier_name), 'rb') as f:
		clf = pickle.load(f)
	#load and process ratio data from csv file
	dfvol = pandas.read_csv(os.path.join(BASE, vol_name))
	df_conn = pandas.read_csv('vikram/dataset.txt')
	df_conn.drop(list(df_conn.filter(regex = '(binary)')), axis = 1, inplace = True)
	df_conn['file_name'] = df_conn['file_name'].str.replace('.trk.gz.AAL2.count.end.network_measures.txt', '.nii.gz')
	dfvol['Scan'] = dfvol['Scan'].str.replace(' ', '')
	dfvol = dfvol.merge(df_conn, left_on='Scan', right_on='file_name')
	predictions_csv = os.path.join(BASE,'predictions.csv')
	f = open(predictions_csv, 'w')
	writer = csv.writer(f)
	for _, corresp_row_ratio in dfvol.iterrows():
		prediction = 'no NPH'
		patient = corresp_row_ratio['Scan']
		vent = corresp_row_ratio['Vent']
		sub = corresp_row_ratio['Sub']
		white = corresp_row_ratio['White']
		density = float(corresp_row_ratio['density'])
		cca = float(corresp_row_ratio['clustering_coeff_average(weighted)'])
		transitivity = float(corresp_row_ratio['transitivity(weighted)'])
		ncpl = float(corresp_row_ratio['network_characteristic_path_length(weighted)'])
		sw = float(corresp_row_ratio['small-worldness(weighted)'])
		ge = float(corresp_row_ratio['global_efficiency(weighted)'])
		dog = float(corresp_row_ratio['diameter_of_graph(weighted)'])
		rog = float(corresp_row_ratio['radius_of_graph(weighted)'])
		assortativity = float(corresp_row_ratio['assortativity_coefficient(weighted)'])
		x = np.array([[vent, sub, white, vent+sub+white, density, cca, transitivity, ncpl, sw, ge, dog, rog, assortativit]]).reshape(1,-1)
		x = preprocessing.scale(x, axis=1)
		predict_num = clf.predict(x)[0]
		if predict_num == 1:
			prediction = 'possible NPH'
		print(prediction)
		writer.writerow([patient, prediction])
	f.close()


def clean_up(BASE):
	'''
	Cleans up files from image processing pipeline.
	'''
	names = ['imname_list.pkl', 'imname_list1.pkl', 'imname_affine.pkl', 'imname_affine1.pkl', 'imname_header.pkl', 'imname_header1.pkl']
	for name in names:
		name = os.path.join(BASE, name)
		if os.path.exists(name):
			os.remove(name)
	


