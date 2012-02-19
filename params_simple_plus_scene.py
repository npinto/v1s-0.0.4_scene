#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" V1S Parameters module

This module describes a model where features from various intermediate
representations are combined with the final outputs. 
"""

import scipy as N

# -- testing protocol
protocol = {
    # number of training examples
    'ntrain':100,
    # number of testing examples
    'ntest':10000,
    # number of trials
    'ntrials':10,
    # random seed
    'seed':1,
    }

# -- representation 
# some filter parameters
norients = 16
orients = [ o*N.pi/norients for o in xrange(norients) ]
divfreqs = [2, 3, 4, 6, 11, 18]
freqs = [ 1./n for n in divfreqs ]
phases = [0]

# dict with all representation parameters
representation = {

# - preprocessing
# prepare images before processing
'preproc': {
    # resize input images by keeping aspect ratio and fix the biggest edge
    'max_edge': 150,
    # kernel size of the box low pass filter
    'lsum_ksize': 3,
    },

# - input local normalization
# local zero-mean, unit-magnitude
'normin': {
    # kernel shape of the local normalization
    'kshape': (3,3),
    # magnitude threshold
    # if the vector's length is below, it doesn't get resized
    'threshold': 1.0,
    },

# - linear filtering
'filter': {
    # kernel shape of the gabors
    'kshape': (43,43),
    # list of orientations
    'orients': orients,
    # list of frequencies
    'freqs': freqs,
    # list of phases
    'phases': phases,
    # threshold (variance explained) for the separable convolution
    'sep_threshold': .9,
    },

# - simple non-linear activation
'activ': {
    # minimum output (clamp)
    'minout': 0,
    # maximum output (clamp)
    'maxout': 1,
    },

# - output local normalization
'normout': {
    # kernel shape of the local normalization
    'kshape': (3,3),
    # magnitude threshold
    # if the vector's length is below, it doesn't get resized
    'threshold': 1.0,
    },

# - dimension reduction
'dimr': {
    # kernel size of the local sum (2d slice)
    'lsum_ksize': 17,
    # fixed output shape (only the first 2 dimensions, y and x)
    'outshape': (30,30),
    },
}

# -- featsel details what features you want to be included in the vector
featsel = {
    # Include representation output ? True or False
    'output': True,

    # Include grayscale values ? None or (height, width)    
    'input_gray': (100,100),
    # Include color histograms ? None or nbins per color
    'input_colorhists': None, 
    # Include input norm histograms ? None or (division, nfeatures)    
    'normin_hists': None,
    # Include filter output histograms ? None or (division, nfeatures)
    'filter_hists': None,
    # Include activation output histograms ? None or (division, nfeatures)    
    'activ_hists': (2,10000),
    # Include output norm histograms ? None or (division, nfeatures)
    'normout_hists': (1,10000),
    # Include representation output histograms ? None or (division, nfeatures)
    'dimr_hists': (1,10000),
    }

# -- model is a list of (representation, featureselection)
# that will be combine resulting in the final feature vector
model = [(representation, featsel)]

# -- pca_threshold will be used to define how much variance
# you want to explain when projecting training/testing data
# to the training eigen subspace
pca_threshold = 1.
