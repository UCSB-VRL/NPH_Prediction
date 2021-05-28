import numpy as np
import os
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import tarfile
import nibabel as nib
import shutil
from skimage.transform import rescale, resize, downscale_local_mean
import ipdb
import pickle
import pandas as pd
import skorch
from skorch import NeuralNetClassifier, NeuralNet, NeuralNetBinaryClassifier
from sklearn.metrics import classification_report
from skorch.helper import SliceDataset
import sys
import models
from models import criterions, unet


def predict_NPH(BASE, gpu):
	dtype = torch.FloatTensor
	if gpu:
		dtype = torch.cuda.FloatTensor 
		device = torch.device("cuda")

	gt_segs = os.listdir(BASE)
	Scan_Folder = os.path.join(BASE, 'Scans')
	scans_all = [s for s in os.listdir(Scan_Folder) if (s.endswith('nii.gz') and not 'MNI152' in s)]

	N = 256
	N1 = 128
	batch_size = 1

	counter = 0

	X = np.empty(0)
	Y = np.empty(0)
	labels = np.empty(0)
	conn_metric = np.empty(0)

	patients = []

	#original range is scans_all
	for j in range(len(scans_all)):
			scanname = scans_all[j]
			scanpath = os.path.join(Scan_Folder, scanname)
			print(scanname)
			conn_metric_path = os.path.join(scanname, 'network_measures.txt')
			conn_metrics = df = pd.read_csv(conn_metric_path, delimiter = "\t", nrows=27)
			if conn_metric.size == 0:
				conn_metric = conn_metrics.iloc[:,1:].to_numpy()
			else:
				conn_metric = np.concatenate((conn_metric, conn_metrics), axis = 0)
			patients.append(scanname)
			img_nib = nib.load(os.path.join(Scan_Folder,scanname))
			img = np.asanyarray(img_nib.dataobj)
			temp_img = resize(img, output_shape=[N,N,N1], preserve_range=True, mode='constant',order=1, anti_aliasing=True)
			temp_img = np.expand_dims(temp_img, axis = 0)
			temp_img = np.expand_dims(temp_img, axis = 0)
			temp_img -= 100
			temp_img /= 100

			if(X.size == 0):
				X = temp_img
			else:
				X = np.concatenate((X, temp_img), axis = 0)


	print(conn_metric.shape)

	print(X.shape)


	def normalization(planes, norm='gn'):
		if norm == 'bn':
			m = nn.BatchNorm3d(planes)
		elif norm == 'gn':
			m = nn.GroupNorm(4, planes)
		elif norm == 'in':
			m = nn.InstanceNorm3d(planes)
		else:
			raise ValueError('normalization type {} is not supported'.format(norm))
		return m


	class ConvD(nn.Module):
		def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
			super(ConvD, self).__init__()

			self.first = first
			self.maxpool = nn.MaxPool3d(2, 2)

			self.dropout = dropout
			self.relu = nn.ReLU(inplace=True)

			self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
			self.bn1   = normalization(planes, norm)

			self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
			self.bn2   = normalization(planes, norm)

			self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
			self.bn3   = normalization(planes, norm)

		def forward(self, x):
			if not self.first:
				x = self.maxpool(x)
			x = self.bn1(self.conv1(x))
			#what's the point of this below
			y = self.relu(self.bn2(self.conv2(x)))
			if self.dropout > 0:
				y = F.dropout3d(y, self.dropout)
			#change y again here
			y = self.bn3(self.conv3(x))
			return self.relu(x + y)


	class ConvU(nn.Module):
		def __init__(self, planes, norm='gn', first=False):
			super(ConvU, self).__init__()

			self.first = first

			if not self.first:
				self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False)
				self.bn1   = normalization(planes, norm)

			self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
			self.bn2   = normalization(planes//2, norm)

			self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
			self.bn3   = normalization(planes, norm)

			self.relu = nn.ReLU(inplace=True)

		def forward(self, x, prev):
			# final output is the localization layer
			if not self.first:
				x = self.relu(self.bn1(self.conv1(x)))

			y = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
			y = self.relu(self.bn2(self.conv2(y)))

			y = torch.cat([prev, y], 1)
			y = self.relu(self.bn3(self.conv3(y)))

			return y



	''' Loading unet '''

	from models import criterions, unet

	unet_model = '/home/angela/NPH/unet_ce_hard_per_im_s8841_all'
	ckpt = 'model_last.tar'

	unet = unet.Unet(do_class = True)
	unet.cuda()


	model_file = os.path.join(unet_model, ckpt)
	checkpoint = torch.load(model_file)
	unet.load_state_dict(checkpoint['state_dict'])

	num_classes = 7
	import copy
	net = copy.deepcopy(unet)
	net.convd1.conv1 = ConvD(1,16,0.5,'gn',first=True)
	net.convd1.conv1.weight = nn.Parameter(unet.convd1.conv1.weight[:,1,:,:,:].unsqueeze(1))
	net.seg3 = nn.Conv3d(128, num_classes, kernel_size=(1,1,1), stride=(1,1,1))
	net.seg2 = nn.Conv3d(64, num_classes, kernel_size=(1,1,1), stride=(1,1,1))
	net.seg1 = nn.Conv3d(32, num_classes, kernel_size=(1,1,1), stride=(1,1,1))
	net.seg3.weight[0:5] = unet.seg3.weight[0:5,:,:,:,:]
	net.seg2.weight[0:5] = unet.seg2.weight[0:5,:,:,:,:]
	net.seg1.weight[0:5] = unet.seg1.weight[0:5,:,:,:,:]
	net.seg3.weight[5:num_classes] = unet.seg3.weight[0:num_classes-5,:,:,:,:]
	net.seg2.weight[5:num_classes] = unet.seg2.weight[0:num_classes-5,:,:,:,:]
	net.seg1.weight[5:num_classes] = unet.seg1.weight[0:num_classes-5,:,:,:,:]
	net.seg1.weight = nn.Parameter(net.seg1.weight)
	net.seg2.weight = nn.Parameter(net.seg2.weight)
	net.seg3.weight = nn.Parameter(net.seg3.weight)


	for param in net.parameters():
		param.requires_grad = False

	#add fully connected layer to end of model

	reload_path = os.path.join(BASE, 'unet_model.pt')
	net.load_state_dict(torch.load(reload_path))
	if gpu:
		net.cuda()
	del unet

	class BinaryClass(nn.Module):
		def __init__(self, net):
			super(BinaryClass, self).__init__()

			self.orig_net = net
			# self.origModel = net
			self.fc1 = nn.Linear(256 * 16 * 16 * 8, 200)
			#hard-coded
			self.fc_con = nn.Linear(26, 26)

			self.fc2 = nn.Linear(226, 1)

		def forward(self, input, conn_data):
			# x = self.origModel(input)
			x = self.orig_net(input)
			x = x.flatten()

			x = F.relu(self.fc1(x))
			x_c = F.relu(self.fc_con(conn_data.flatten()))
			x = torch.cat((x, x_c), 0)
			#remember to use threshold at end
			x = self.fc2(x)

			return x


	for i in range(X.shape[0]):
		X_test = X[i]
		conn_test = conn_metric[i][:,:-1].astype(float)
		model_class = BinaryClass(net)
		if gpu:
			model_class.cuda()

		model_class.eval()
		test_outputs = np.empty(0)

		test_x = torch.Tensor(X_test[i, : ,: ,: ,:][None,:]).to(device)
		test_conn = torch.Tensor(conn_test[i,:][None,:]).to(device)

		with torch.no_grad():
			output = model_class(test_x, test_conn)

		output_thresh = output
		output_thresh[output_thresh >= 0.5] = 1
		output_thresh[output_thresh < 0.5] = 0
		output_thresh = output_thresh.detach().cpu().numpy()
		
		if(test_outputs.size == 0):
			test_outputs = np.asarray([output_thresh])
		else:
			test_outputs = np.concatenate((test_outputs, [output_thresh]), axis = 0)

