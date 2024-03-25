#!/bin/bash

echo "\n\n\nBENCHMARK\n\n\n"
python train.py

echo "\n\n\nMODEL HEAD\n\n\n"
python train.py model.detector_depthwise=false
python train.py model.detector_depthwise=false model.detector_head=false

echo "\n\n\nKERNEL SIZE\n\n\n"
python train.py model.t_kernel_size=3

echo "\n\n\nEVENT PROCESSING METHOD\n\n\n"
python train.py dataset.spatial_affine=false
python train.py dataset.temporal_flip=false
python train.py dataset.temporal_scale=false

echo -e "\n\n\FULL CONV 3D\n\n\n"
python train.py model.full_conv3d=true

echo -e "\n\n\nEVENT INTERPOLATION\n\n\n"
python train.py dataset.events_interpolation=bilinear

#echo -e "\n\n\nNORMS\n\n\n"
#python train.py model.norms=allbn
#python train.py model.norms=allgn
