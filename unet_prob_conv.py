import numpy as np
import os
import copy
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
import torch.optim as optim
import tarfile
import nibabel as nib
import shutil
from skimage.transform import rescale, resize, downscale_local_mean
import ipdb
from sklearn.model_selection import KFold
import pickle
import pdb
# from numba import jit, vectorize, int32

from torchvision.datasets.folder import DatasetFolder
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import models 
from models.unet_vikram_prob_conv import Unet
from models import criterions

torch.backends.cudnn.benchmark = True



dtype = torch.FloatTensor
do_yellow = False
GT_Folder = 'UCISegmentations'
Scan_Folder = 'Scans'


do_prep = False
N = 256
N1 = 128
batch_size = 1
num_epochs = 500
start_k = 0
reload_k_epoch = False
reload_k = 4
reload_l = 175
lr = 0.001
num_classes = 7
use_amp = True


kfold = KFold(shuffle=True)
#crit = 'weighted_hard'
crit = 'hard_per_im'




class NPHDataset(Dataset):
    """
    
    +----------------------------------------------------------------------------------+    
    
        Normal Pressure Hydrocephalus (NPH) Dataset
        Authors/Maintainers: Angela Zhang, Vikram [James], Amil Khan


        Designed for loading CT scans for predicting NPH, this function

        - Lists everything in the ground truth directory
        - Gets all of the Scans from Scans Folder (.nii.gz)
        - Gets ground truth segmented images from ground truth folder (.nii)

        
        Parameters:
        -----------
        Scan_Folder : String 
        Directory with all the Scans
        
        GT_Folder : String
        Directory with Ground Truth


        Returns
        -------
        Image Info  : Tensor

        Input   (X) : Tensor

        Label   (Y) : Tensor
        
        Weights (Z) : Tensor
                
        Scan Name   : String
        
    
    +----------------------------------------------------------------------------------+
    """

    def __init__(self, Scan_Folder, GT_Folder, transform=None):

        gt_segs = os.listdir(GT_Folder)
        self.scans_all = [s for s in os.listdir(Scan_Folder) if (s.endswith('nii.gz') and not 'MNI152' in s)]
        self.get_segs = [s[6:] for s in gt_segs if (not os.path.isdir(os.path.join(GT_Folder, s))) and ('.nii' in s)]

        
    def __len__(self):
        return len(self.get_segs)
    
    
    def getDistance(self, segarray, nclasses=3, transform_type='edt'):
        """
        Get Distance 


        Parameters
        ----------
        Segmentation Array : Numpy Array

        Number of Classes  : Int

        Transform Type     : default='edt'


        Returns
        -------
        Distance Matrix : Numpy Array

        """
        shape = segarray.shape
        dist_mtx = np.ones((nclasses, shape[0], shape[1], shape[2]))
        return dist_mtx
   

    def get_img_and_seg(self, gt_seg):
        ''' 
        Get the Image and Segmentation for input into the network 

        
        Parameters
        ----------
        Filepath (gt_seg) : Str


        Returns
        -------
        Image Info  : Tensor

        Input   (X) : Tensor

        Label   (Y) : Tensor
        
        Weights (Z) : Tensor
                
        Scan Name   : String

        '''
        X = np.empty([1,N,N,N1])
        Y = np.empty([N,N,N1])
        Z = np.empty([1,3,N,N,N1])
        patient = gt_seg
        if '_20' in patient:
            scanname = gt_seg
        else:
            gt_seglist = gt_seg.split('.')
            patient = gt_seglist[0]
            #ipdb.set_trace()
            if len(gt_seglist) > 5:
                scan = str(int(gt_seglist[2]))
                subscan = str(int(gt_seglist[4]))
            else:
                if len(gt_seglist) < 4:
                    ipdb.set_trace()
                scan = gt_seglist[1][4:]
                subscan = gt_seglist[2][7:]
            scanname = None
            for name in self.scans_all:
                namelist = name.split('.')
                patient1 = namelist[0]
                scan1 = namelist[1]
                subscan1 = namelist[2]
                if patient==patient1 and scan==scan1 and subscan==subscan1:
                    scanname = name
            if scanname is None:
                print('could not find scan from gt seg')
                return
        img_nib = nib.load(os.path.join(Scan_Folder,scanname))
        seg_nib = nib.load(os.path.join(GT_Folder,'Final_'+gt_seg))
        img = np.asanyarray(img_nib.dataobj)
        seg = np.asanyarray(seg_nib.dataobj)
        img_info = (img.shape, img_nib.affine)
        temp_img = resize(img, output_shape=[N,N,N1], preserve_range=True, mode='constant',order=1, anti_aliasing=True)
        temp_seg = resize(seg, output_shape=[N,N,N1], preserve_range=True, mode='constant',order=0, anti_aliasing=False).astype(int).astype(float)
        temp_weight = self.getDistance(temp_seg)
        X[0] = temp_img
#         print(X.shape)
        Y = temp_seg
#         print(Y.shape)
        Z = temp_weight
#         print(Z.shape)
        X -= 100
        X /= 100
        #X -= np.mean(X)
        #X /= np.std(X)
        return {"img_info":img_info, "inputs":torch.from_numpy(X).type(dtype), "labels":torch.from_numpy(Y).type(dtype).long(), "weights": torch.from_numpy(Z).type(dtype), "scan_name":scanname}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  

        return self.get_img_and_seg(self.get_segs[idx])



nph_dataset = NPHDataset(Scan_Folder, GT_Folder)


unet_model = 'unet_ce_hard_per_im_s8841_all'
ckpt = 'model_last.tar'

unet = Unet()
unet.cuda()

#model_file = os.path.join(unet_model, ckpt)
#checkpoint = torch.load(model_file)

#Changes the weight shape in the state dict to account for the additional heatmaps
test = torch.rand([32,34,3,3,3])
#test[:,0:32,:,:,:] = checkpoint['state_dict']['convu1.conv3.weight']
#checkpoint['state_dict']['convu1.conv3.weight'] = test

#unet.load_state_dict(checkpoint['state_dict'], strict = False)
nn.init.uniform_(unet.prob_conv.conv1.weight)
nn.init.uniform_(unet.prob_conv.bn1.weight)
nn.init.uniform_(unet.prob_conv.bn1.bias)
nn.init.uniform_(unet.prob_conv.conv2.weight)
nn.init.uniform_(unet.prob_conv.bn2.weight)
nn.init.uniform_(unet.prob_conv.bn2.bias)
print('Weights initialized')
#loads weights 
#ipdb.set_trace()

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

# Define Loss Criterion
if crit == 'hard_per_im':
	criterion = criterions.hard_per_im_cross_entropy
elif crit == 'hard':
	criterion = criterions.hard_cross_entropy
elif crit == 'weighted_BCE':
	criterion = criterions.weighted_BCE
elif crit == 'weighted_hard':
	criterion = criterions.weighted_hard

# Vikram's masterpiece I think
class ConvD(nn.Module):
	def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
		super(ConvD, self).__init__()

		self.first = first
		self.maxpool = nn.MaxPool3d(2, 2)

		self.dropout = dropout
		self.relu = nn.ReLU(inplace=True)

		self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
		self.bn1 = normalization(planes, norm)

		self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
		self.bn2 = normalization(planes, norm)

		self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
		self.bn3 = normalization(planes, norm)
    
	def forward(self, x):
		if not self.first:
			x = self.maxpool(x)
		x = self.bn1(self.conv1(x))
		y = self.relu(self.bn2(self.conv2(x)))
		if self.dropout > 0:
			y = F.dropout3d(y, self.dropout)
		y = self.bn3(self.conv3(x))
		return self.relu(x + y)

net = copy.deepcopy(unet)
net.convd1.conv1 = ConvD(1,16,0.5,'gn',first=True)
net.convd1.conv1.weight = nn.Parameter(unet.convd1.conv1.weight[:,1,:,:,:].unsqueeze(1))
net.seg3 = nn.Conv3d(128, num_classes, kernel_size=(1,1,1), stride=(1,1,1))
net.seg2 = nn.Conv3d(64, num_classes, kernel_size=(1,1,1), stride=(1,1,1))
net.seg1 = nn.Conv3d(32, num_classes, kernel_size=(1,1,1), stride=(1,1,1))
#ipdb.set_trace()
net.seg3.weight.data[0:5] = unet.seg3.weight.data[0:5,:,:,:,:]
net.seg2.weight.data[0:5] = unet.seg2.weight.data[0:5,:,:,:,:]
net.seg1.weight.data[0:5] = unet.seg1.weight.data[0:5,:,:,:,:]
net.seg3.weight.data[5:num_classes] = unet.seg3.weight.data[0:num_classes-5,:,:,:,:]
net.seg2.weight.data[5:num_classes] = unet.seg2.weight.data[0:num_classes-5,:,:,:,:]
net.seg1.weight.data[5:num_classes] = unet.seg1.weight.data[0:num_classes-5,:,:,:,:]
net.seg1.weight = nn.Parameter(net.seg1.weight)
net.seg2.weight = nn.Parameter(net.seg2.weight)
net.seg3.weight = nn.Parameter(net.seg3.weight)
del unet

# Run on multiple GPUs, should use DistributedDataParallel instead but whatevs
net = torch.nn.DataParallel(net)

def get_mask_from_output(output, info):
	n,c = output.shape[:2]
	N = output.shape[3]
	output = output.view(n,c,-1)
	lsoftmax = nn.LogSoftmax(dim=1)
	output = lsoftmax(output)
	output = output.argmax(dim=1)
	if n == 1:
		output = output.view(N,N,N1)
	else:
		output = output.view(n,N,N,N1)
	output = output.cpu().detach().numpy()
	#output = resize(output, info[0], preserve_range=True, mode='constant', order=0)
	return output


def dice_score(ml_segarray, gt_segarray):
    """
    Computes the Dice Score


    Parameters
    ----------
    ml_segarray : Numpy Array

    gt_segarray : Numpy Array


    Returns
    -------
    dice_vent : Float

    dice_wm   : Float

    dice_sub  : Float
    """
    ml_vent = np.zeros(ml_segarray.shape)
    ml_vent[np.where(ml_segarray == 1)] = 1
    gt_vent = np.zeros(gt_segarray.shape)
    if gt_vent.shape != gt_segarray.shape or gt_vent.shape != ml_vent.shape:
        ipdb.set_trace()
    gt_vent[np.where(gt_segarray == 1)] = 1
    TP_vent = np.sum(np.logical_and(gt_vent, ml_vent))
    gt_no_vent = 1 - gt_vent
    FP_vent = np.sum(np.logical_and(gt_no_vent, ml_vent))
    ml_no_vent = 1 - ml_vent
    TN_vent = np.sum(np.logical_and(gt_no_vent, ml_no_vent))
    FN_vent = np.sum(np.logical_and(gt_vent, ml_no_vent))

    ml_wm = np.zeros(ml_segarray.shape)
    ml_wm[np.where(ml_segarray == 2)] = 1
    gt_wm = np.zeros(gt_segarray.shape)
    gt_wm[np.where(gt_segarray == 2)] = 1
    TP_wm = np.sum(np.logical_and(ml_wm,gt_wm))
    gt_no_wm = 1 - gt_wm
    FP_wm = np.sum(np.logical_and(gt_no_wm, ml_wm))
    ml_no_wm = 1 - ml_wm
    TN_wm = np.sum(np.logical_and(gt_no_wm, ml_no_wm))
    FN_wm = np.sum(np.logical_and(gt_wm, ml_no_wm))

    ml_sub = np.zeros(ml_segarray.shape)
    ml_sub[np.where(ml_segarray == 3)] = 1
    gt_sub = np.zeros(gt_segarray.shape)
    gt_sub[np.where(gt_segarray == 3)] = 1
    TP_sub = np.sum(np.logical_and(ml_sub,gt_sub))
    gt_no_sub = 1 - gt_sub
    FP_sub = np.sum(np.logical_and(gt_no_sub, ml_sub))
    ml_no_sub = 1 - ml_sub
    TN_sub = np.sum(np.logical_and(gt_no_sub, ml_no_sub))
    FN_sub = np.sum(np.logical_and(gt_sub, ml_no_sub))

    dice_vent = TP_vent / np.mean([np.sum(gt_vent), np.sum(ml_vent)])
    dice_wm = TP_wm / np.mean([np.sum(gt_wm), np.sum(ml_wm)])
    dice_sub = TP_sub / np.mean([np.sum(gt_sub), np.sum(ml_sub)])
    if dice_vent > 1 or dice_wm > 1 or dice_sub > 1:
        ipdb.set_trace()
    return dice_vent, dice_wm, dice_sub


# Writer will output to ./runs/ directory by default
#writer = SummaryWriter(comment='-prob_conv')


for fold, (train_ids, test_ids) in enumerate(kfold.split(nph_dataset)):
    

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(comment=f'-prob_conv-FOLD-{fold}')
    
    # Print current Fold
    print('='*50)
    print(f'FOLD {fold}')
    print('='*50)
    

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define dataloaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      nph_dataset, num_workers=20, pin_memory=True, batch_size=batch_size,
                      sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      nph_dataset, num_workers=8, pin_memory=True, 
                      batch_size=batch_size, sampler=test_subsampler)

    print("---> Successfully Loaded Trainloader and Testloader...")
    
    # Init the neural network
    net.to('cuda')
    
    print("---> Successfully Initialized Network on GPU...")
    
    # Initialize optimizer
    lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=True, weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    
    loss_track_train    = []
    loss_track_test     = []
    dice_vent_train     = []
    dice_cerebral_train = []
    dice_sub_train      = []
    dice_vent_test      = []
    dice_cerebral_test  = []
    dice_sub_test       = []
    epochs_per_score    = []
    
    
    
    
    # Run the training loop for defined number of epochs
    for l in range(num_epochs):
        
        # Record Average Train Loss and Average Dice Scores
        avg_loss_train          = 0.0
        avg_loss_test           = 0.0
        avg_dice_vent_train     = 0.0
        avg_dice_cerebral_train = 0.0
        avg_dice_sub_train      = 0.0
        avg_dice_vent_test      = 0.0
        avg_dice_cerebral_test  = 0.0
        avg_dice_sub_test       = 0.0
        

        # Print epoch
        print(f'\nStarting Epoch {l+1}')
        print('-'*50)

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get the inputs
            # _, inputs, labels, weights, scan_name = data
            
            inputs    = data['inputs'].to('cuda', non_blocking=True)
            scan_name = data['scan_name'][0]
            labels    = data['labels'].to('cuda', non_blocking=True)

            # Runs the forward pass under autocast    
            with autocast(enabled=use_amp):
            
                # Perform Forward Pass    
                outputs = net(inputs, scan_name)
                # assert outputs.dtype is torch.float16
            
                # Compute Loss
                if crit.startswith('weighted'):
                    loss = criterion(outputs, labels, weights)
                else:
                    loss = criterion(outputs, labels)
            # Print statistics
            current_loss += loss.item()
            print("Current Loss: ", current_loss)
            

            # Perform Backward Pass
            scaler.scale(loss).backward()
            
            # Perform Optimization
            scaler.step(optimizer)
            scaler.update()
            
            # Zero the Gradient and set to None
            optimizer.zero_grad(set_to_none=True)
            print(f'Done with {i}')

        # Update Loss every Epoch
        writer.add_scalar('Loss', current_loss, l)

                
        if l%5==0 and (l > 20):
            torch.save(net.state_dict(), 'trials/kfold_'+str(fold)+'_unet_drop.5_lr0.001_tv_w1_PT_hard_epoch_' + str(l)+'.pt')

        if l%20==0: 
            net.eval()
            with torch.no_grad():
                num_train = len(train_ids)
                for i, data in enumerate(trainloader, 0):
                    #info, inputs, labels, weights, scan_name = data
                    inputs    = data['inputs'].to('cuda', non_blocking=True)
                    scan_name = data['scan_name'][0]
                    labels    = data['labels'].to('cuda', non_blocking=True)
                    
                    outputs = net(inputs,scan_name)
                    
                    if crit.startswith('weighted'):
                        loss = criterion(outputs, labels, weights)
                    else:
                        loss = criterion(outputs, labels)
                    avg_loss_train += loss.item()
                    #dice
                    outputs = get_mask_from_output(outputs, data['img_info'])
                    dice_vent, dice_wm, dice_sub = dice_score(outputs, np.squeeze(labels.cpu().detach().numpy()))
                    avg_dice_vent_train += dice_vent
                    avg_dice_cerebral_train += dice_wm
                    avg_dice_sub_train += dice_sub

                    writer.add_scalar('Ventricle Dice Score', dice_vent, i)
                    writer.add_scalar('Cerebral Dice Score', dice_wm, i)
                    writer.add_scalar('Subcortical Dice Score', dice_sub, i)

                loss_track_train.append(avg_loss_train/(i+1))
                dice_cerebral_train.append(avg_dice_cerebral_train/(i+1))
                dice_vent_train.append(avg_dice_vent_train/(i+1))
                dice_sub_train.append(avg_dice_sub_train/(i+1))
                
                if avg_dice_cerebral_train/(i+1) > 1:
                    ipdb.set_trace()


                print(f'\nFOLD {fold} | TRAIN LOSS SUMMARY')
                print('-'*50)
                print(
                    f' Train: Average Loss            {avg_loss_train/(i+1)}\n',
                    f'Train: Cerebral Dice Score     {avg_dice_cerebral_train/(i+1)}\n',
                    f'Train: Ventricle Dice Score    {avg_dice_vent_train/(i+1)}\n',
                    f'Train: Subcortical Dice Score  {avg_dice_sub_train/(i+1)}',
                )
                print('-'*50)


                writer.add_scalar('Train: Average Loss', avg_loss_train/(i+1), l)
                writer.add_scalar('Train: Average Cerebral Dice Score', avg_dice_cerebral_train/(i+1), l)
                writer.add_scalar('Train: Average Subcortical Dice Score', avg_dice_sub_train/(i+1), l)
                writer.add_scalar('Train: Ventricle Dice Score', avg_dice_vent_train/(i+1), l)


                num_test = len(test_ids)
                for i, data in enumerate(testloader, 0):
                    # info, inputs, labels, weights, scan_name = data
                    inputs    = data['inputs'].to('cuda', non_blocking=True)
                    scan_name = data['scan_name'][0]
                    labels    = data['labels'].to('cuda', non_blocking=True)
                    
                    outputs = net(inputs, scan_name)
                    
                    if crit.startswith('weighted'):
                        loss = criterion(outputs, labels, weights)
                    else:
                        loss = criterion(outputs, labels)
                    avg_loss_test += loss.item()
                    # Dice Score
                    outputs = get_mask_from_output(outputs, data['img_info'])
                    dice_vent, dice_wm, dice_sub = dice_score(outputs, np.squeeze(labels.cpu().detach().numpy()))
                    avg_dice_vent_test += dice_vent
                    avg_dice_cerebral_test += dice_wm
                    avg_dice_sub_test += dice_sub
                loss_track_test.append(avg_loss_test/(i+1))
                dice_vent_test.append(avg_dice_vent_test/(i+1))
                dice_cerebral_test.append(avg_dice_cerebral_test/(i+1))
                dice_sub_test.append(avg_dice_sub_test/(i+1))
                epochs_per_score.append(l)

                writer.add_scalar('Test: Average Loss', avg_loss_test/(i+1), l)
                writer.add_scalar('Test: Average Cerebral Dice Score', avg_dice_cerebral_test/(i+1), l)
                writer.add_scalar('Test: Average Subcortical Dice Score', avg_dice_sub_test/(i+1), l)
                writer.add_scalar('Test: Ventricle Dice Score', avg_dice_vent_test/(i+1), l)


                print(f'\nFOLD {fold} | TEST LOSS SUMMARY')
                print('-'*50)
                print(
                    f' Test: Average Loss            {avg_loss_test/(i+1)}\n',
                    f'Test: Cerebral Dice Score     {avg_dice_cerebral_test/(i+1)}\n',
                    f'Test: Ventricle Dice Score    {avg_dice_vent_test/(i+1)}\n',
                    f'Test: Subcortical Dice Score  {avg_dice_sub_test/(i+1)}',
                )
                print('-'*50)


    np.save('scores/w1_PT_hard_loss_train'+str(fold)+'.npy', loss_track_train)
    np.save('scores/w1_PT_hard_loss_test'+str(fold)+'.npy', loss_track_test)
    np.save('scores/w1_PT_hard_dice_vent_train'+str(fold)+'.npy', dice_vent_train)
    np.save('scores/w1_PT_hard_dice_vent_test'+str(fold)+'.npy', dice_vent_test)
    np.save('scores/w1_PT_hard_dice_cerebral_train'+str(fold)+'.npy', dice_cerebral_train)
    np.save('scores/w1_PT_hard_dice_cerebral_test'+str(fold)+'.npy', dice_cerebral_test)
    np.save('scores/w1_PT_hard_dice_sub_train'+str(fold)+'.npy', dice_sub_train)
    np.save('scores/w1_PT_hard_dice_sub_test'+str(fold)+'.npy', dice_sub_test)
    np.save('scores/w1_PT_hard_epochs_per_score'+str(fold)+'.npy', epochs_per_score)

print('='*50)
print('='*50)
print('---> TRAINING COMPLETE! <---')
print('='*50)
print('='*50)









