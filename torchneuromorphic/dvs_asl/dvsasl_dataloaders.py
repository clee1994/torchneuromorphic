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
import struct
import time
import numpy as np
import scipy.misc
import h5py
import torch.utils.data
from .create_hdf5 import create_events_hdf5
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from ..transforms import *
import os

mapping = { 0 :'A',
            1 :'B',
            2 :'C',
            3 :'D',
            4 :'E',
            5 :'F',
            6 :'G',
            7 :'H',
            8 :'I',
            9 :'K',
            10:'L',
            11:'M',
            12:'N',
            13:'O',
            14:'P',
            15:'Q',
            16:'R',
            17:'S',
            18:'T',
            19:'U',
            20:'V',
            21:'W',
            22:'X',
            23:'Y'
            }


class DVSASLDataset(NeuromorphicDataset):

    def __init__(
            self, 
            root,
            train=True,
            transform=None,
            target_transform=None,
            download_and_create=True,
            chunk_size = 100,
            nclasses = 5,
            samples_per_class = 2,
            labels_u = range(5)):


        self.directory = "/".join(root.split("/")[:-1])+"/"
        self.resources_url = [['Manually Download dataset here: https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0 and place under {0}'.format(self.directory), None, 'ICCV2019_DVS_dataset.zip']]
        self.resources_local = [self.directory+'raw']

        self.n = samples_per_class * nclasses
        self.nclasses = nclasses

        self.download_and_create = download_and_create
        self.root = root
        self.train = train 
        self.chunk_size = chunk_size

        self.labels_u = labels_u
        self.labels = np.repeat(self.labels_u, samples_per_class)
        self.labels_map =  dict(zip(np.unique(self.labels),np.arange(nclasses)))


        super(DVSASLDataset, self).__init__(
                root,
                transform=transform,
                target_transform=target_transform,
                extract_root = self.directory+'raw/')

        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            if train:
                self.keys = f['extra']['train_keys'][()]
                self.keys_by_label = f['extra']['train_keys_by_label'][()]
            else:
                self.keys = f['extra']['test_keys'][()]
                self.keys_by_label = f['extra']['test_keys_by_label'][()]

        self.keys = np.array([ np.random.choice(s, samples_per_class) for s in self.keys_by_label[self.labels_u]]).reshape(-1)


    def download(self):
        super(DVSASLDataset, self).download()

    def create_hdf5(self):
        create_events_hdf5(self.resources_local[0], self.root)

    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        # #Important to open and close in getitem to enable num_workers>0
        # with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
        #     if not self.train:
        #         key = key + f['extra'].attrs['Ntrain']
        #     import pdb; pdb.set_trace()
        #     assert key in self.keys
        #     data, target = sample(
        #             f,
        #             key,
        #             T = self.chunk_size,
        #             shuffle=self.train)


        #Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:

            if self.train:
                key = f['extra']['train_keys'][self.keys[key]]
            else:
                key = f['extra']['test_keys'][self.keys[key]]
            data, target = sample(
                    f,
                    key,
                    T = self.chunk_size)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return data, target

def sample(hdf5_file,
        key,
        T = 300):
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    tend = dset['times'][-1] 
    start_time = 0

    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad[:,0]-=tmad[0,0]
    return tmad, label

 
def create_datasets(
        root = 'data/dvsasl/dvsasl.hdf5',
        batch_size = 72 ,
        chunk_size_train = 100,
        chunk_size_test = 100,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        nclasses = 5,
        samples_per_class = 1,
        samples_per_test = 256,
        classes_meta = np.arange(14, dtype='int')):

    # this line needs to be corrected!
    size = [2, (179//ds)+1, (179//ds)+1]

    if transform_train is None:
        transform_train = Compose([
            CropDims(low_crop=[30,0], high_crop=[209,179], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_train, size = size),
            ToTensor()])
    if transform_test is None:
        transform_test = Compose([
            CropDims(low_crop=[30,0], high_crop=[209,179], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_test, size = size),
            ToTensor()])
    if target_transform_train is None:
        target_transform_train = Compose([Repeat(chunk_size_train), toOneHot(nclasses)])
    if target_transform_test is None:
        target_transform_test = Compose([Repeat(chunk_size_test), toOneHot(nclasses)])


    labels_u = np.random.choice(classes_meta, nclasses,replace=False) #100 here becuase we have two pairs of digits between 0 and 9

    train_ds = DVSASLDataset(root, train=True,
                                 transform = transform_train, 
                                 target_transform = target_transform_train, 
                                 chunk_size = chunk_size_train,
                                 nclasses = nclasses,
                                 samples_per_class = samples_per_class,
                                 labels_u = labels_u)

    test_ds = DVSASLDataset(root, transform = transform_test, 
                                 target_transform = target_transform_test, 
                                 train=False,
                                 chunk_size = chunk_size_test,
                                 nclasses = nclasses,
                                 samples_per_class = samples_per_test,
                                 labels_u = labels_u)

    return train_ds, test_ds


def create_dataloader(
        root = 'data/dvsasl/dvsasl.hdf5',
        batch_size = 72,
        chunk_size_train = 100,
        chunk_size_test = 100,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        nclasses = 5,
        samples_per_class = 1,
        samples_per_test = 256,
        classes_meta = np.arange(14, dtype='int'),
        batch_size_test = None,
        **dl_kwargs):


    train_d, test_d = create_datasets(
        root = root,
        batch_size = batch_size,
        chunk_size_train = chunk_size_train,
        chunk_size_test = chunk_size_test,
        ds = ds,
        dt = dt,
        transform_train = transform_train,
        transform_test = transform_test,
        target_transform_train = target_transform_train,
        target_transform_test = target_transform_test,
        classes_meta = classes_meta,
        nclasses = nclasses,
        samples_per_class = samples_per_class,
        samples_per_test = samples_per_test)


    train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, **dl_kwargs)
    test_dl = torch.utils.data.DataLoader(test_d, shuffle=True, batch_size=batch_size_test, **dl_kwargs)

    return train_dl, test_dl


def sample_dvsasl_task( N = 5,
                        K = 1,
                        K_test = 256,
                        meta_split = [range(14), range(14,20), range(20,24)],
                        meta_dataset_type = 'train',
                        batch_size_test = None,
                        **kwargs):


    if batch_size_test == None:
        batch_size_test = batch_size

    classes_meta = {}
    classes_meta['train'] = np.array(meta_split[0], dtype='int')
    classes_meta['val']   = np.array(meta_split[1], dtype='int')
    classes_meta['test']  = np.array(meta_split[2], dtype='int')

    assert meta_dataset_type in ['train', 'val', 'test']
    return create_dataloader(classes_meta = classes_meta[meta_dataset_type], nclasses= N, samples_per_class = K, samples_per_test = K_test, batch_size_test = batch_size_test, **kwargs)

