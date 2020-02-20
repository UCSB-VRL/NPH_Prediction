#!/usr/bin/env python2
# -*- coding: utf-8 -*-

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

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:02:52 2017

@author: pkao
"""

import os
import CTtools
from subprocess import call
import sys


ct_scan_path = str(sys.argv[1])

MNI_152_bone = os.path.join(os.getcwd(),'MNI152_T1_1mm_bone.nii.gz')

MNI_152 = os.path.join(os.getcwd(),'MNI152_T1_1mm.nii.gz')

bspline_path = os.path.join(os.getcwd(), 'Par0000bspline.txt')

nameOfAffineMatrix = ct_scan_path[:ct_scan_path.find('.nii.gz')]+'_affine.mat'

#print('The location of MNI152 bone:' , MNI_152_bone)

# if you want to do ct scan removal 
#ct_scan_wodevice = CTtools.removeCTscandevice(ct_scan_path)

ct_scan_wodevice = ct_scan_path

try:

	ct_scan_wodevice_bone = CTtools.bone_extracted(ct_scan_wodevice)

	call(['flirt', '-in', ct_scan_wodevice_bone, '-ref', MNI_152_bone, '-omat', nameOfAffineMatrix, '-bins', '256', '-searchrx', '-180', '180', '-searchry', '-180', '180', '-searchrz', '-180', '180', '-dof', '12', '-interp', 'trilinear'])

	call(['flirt', '-in', ct_scan_wodevice, '-ref', MNI_152, '-applyxfm', '-init', nameOfAffineMatrix, '-out', ct_scan_wodevice[:ct_scan_wodevice.find('.nii.gz')]+'_MNI152.nii.gz'])

except:
	print('something did not work in CT2MNI152 for this image.')


