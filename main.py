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

import sys
import argparse
import os
from imUtils import *
from connectometry import *
from postUtils import *
from unetSeg import *
from predict_NPH import *


def main(base, use_heatmap, gpu, predict_nph, save_last, clear_cache):
	if clear_cache:
		from subprocess import call
		call(['rm', '-r', os.path.join(base, 'UNet_Outputs')])
		call(['rm', '-r', os.path.join(base, 'MNI152')])
		call(['rm', '-r', os.path.join(base, 'imname_list.pkl')])
		call(['rm', '-r', os.path.join(base, 'imname_list1.pkl')])

	else:
		if use_heatmap or predict_nph:
			affine_transform(base) #done
			reverse_transform(base) #done needs testing
		if use_heatmap:
			get_heatmaps(base) #done needs testing
			unet_prob_conv(base, gpu) #Amil: make this take scans from 'Scans' folder, heatmaps from 'Transformed_Heatmaps' folder, and output segmentation in 'UNet_Outputs' folder. if gpu=True, use cuda
		else:
			unetPredict(base, gpu) #done
		if predict_nph:
			transform_seg(base) #done
			get_ventricles(base) #done
			run_connectometry(base) #done needs testing
			predict_NPH(base, gpu)
		clean_up(base)
	
  
if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--directory', default='')
	parser.add_argument('--use_heatmap', action='store_true', default=False)
	parser.add_argument('--gpu', action='store_true', default=True)
	parser.add_argument('--save_last', action='store_true', default=False, help='include this to append to previous csv analysis files')
	parser.add_argument('--clear_cache', action='store_true', default=False, help='this will delete previous calculations')
	parser.add_argument('--predict_nph', action='store_true', default=True)
	args = parser.parse_args()
	main(args.directory, args.use_heatmap, args.gpu, args.predict_nph, args.save_last, args.clear_cache)


