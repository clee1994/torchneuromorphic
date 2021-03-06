#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Author: Clemens JS Schaefer
#
# Creation Date : Tue 01 Sep 2020 10:05:17 PM PST
# Last Modified : Tue 01 Sep 2020 10:05:17 PM PST
#
# Copyright : CJS Schaefer, University of Notre Dame (c)
# Licence : Apache License, Version 2.0
#-----------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm
import scipy.misc
import h5py
import glob
import torch.utils.data
from torchvision.datasets.utils import extract_archive
from ..events_timeslices import *
from ..utils import *
import os
import scipy.io as sio


mapping = { 'a':0,
            'b':1,
            'c':2,
            'd':3,
            'e':4,
            'f':5,
            'g':6,
            'h':7,
            'i':8,
            'k':9,
            'l':10,
            'm':11,
            'n':12,
            'o':13,
            'p':14,
            'q':15,
            'r':16,
            's':17,
            't':18,
            'u':19,
            'v':20,
            'w':21,
            'x':22,
            'y':23
            }

def create_events_hdf5(directory, hdf5_filename):
    # 80/20 train/test
    fns_train = []
    fns_test  = []

    # unzip all zips
    for file in os.listdir(directory):
        if file.endswith(".zip"):
            print("Extracting: {}".format(file))
            extract_archive(os.path.join(directory, file), directory, False)
            for file_sub in os.listdir(os.path.join(directory, file.split(' ')[-1][0])):
                if file_sub.endswith(".mat"):
                    if int(file_sub.split('_')[-1].split('.')[0]) < 3361:
                        fns_train.append(os.path.join(directory, file.split(' ')[-1][0], file_sub))
                    else:
                        fns_test.append(os.path.join(directory, file.split(' ')[-1][0], file_sub))
    #sio.loadmat(file_path)

    test_keys = []
    train_keys = []
    train_label_list = [[] for i in range(24)]
    test_label_list = [[] for i in range(24)]

    #assert len(fns_train)==98

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()

        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        for file_d in tqdm(fns_train+fns_test):
            istrain = file_d in fns_train

            data = sio.loadmat(file_d)

            label = mapping[file_d.split('/')[-2]]

            #data, labels_starttime = aedat_to_events(file_d)
            #tms = data[:,0]
            #ads = data[:,1:]
            #lbls = labels_starttime[:,0]
            #start_tms = labels_starttime[:,1]
            #end_tms = labels_starttime[:,2]
            #out = []

            if istrain: 
                train_keys.append(key)
                train_label_list[label].append(key)
            else:
                test_keys.append(key)
                test_label_list[label].append(key)
            #s_ = get_slice(tms, ads, start_tms[i], end_tms[i])
            #times = s_[0]
            #addrs = s_[1]
            #subj, light = file_d.split('/')[-1].split('.')[0].split('_')[:2]
            metas.append({'key':str(key), 'training sample':istrain}) # 'subject':subj,'light condition':light,
            subgrp = data_grp.create_group(str(key))
            tm_dset = subgrp.create_dataset('times' , data=data['ts'], dtype=np.uint32)
            ad_dset = subgrp.create_dataset('addrs' , data=np.concatenate((data['pol'], data['x'], data['y']), axis=1), dtype=np.uint8)
            lbl_dset= subgrp.create_dataset('labels', data=label, dtype=np.uint8)
            subgrp.attrs['meta_info'] = str(metas[-1])
            #assert lbls[i]-1 in range(11)
            key += 1
        extra_grp.create_dataset('train_keys', data=train_keys)
        extra_grp.create_dataset('test_keys', data=test_keys)
        extra_grp.create_dataset('train_keys_by_label', data = train_label_list)
        extra_grp.create_dataset('test_keys_by_label', data = test_label_list)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)
            
# def gather_aedat(directory, start_id, end_id, filename_prefix = 'user'):
#     if not os.path.isdir(directory):
#         raise FileNotFoundError("DVS Gestures Dataset not found, looked at: {}".format(directory))
#     import glob
#     fns = []
#     for i in range(start_id,end_id):
#         search_mask = directory+'/'+filename_prefix+"{0:02d}".format(i)+'*.aedat'
#         glob_out = glob.glob(search_mask)
#         if len(glob_out)>0:
#             fns+=glob_out
#     return fns


