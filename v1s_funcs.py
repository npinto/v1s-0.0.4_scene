#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" v1s_funcs module

Key sub-operations performed in a simple V1-like model 
(normalization, linear filtering, downsampling, etc.)

"""

import Image
import scipy as N
import scipy.signal

conv = scipy.signal.convolve

# -------------------------------------------------------------------------
def v1s_norm(hin, kshape, threshold):
    """ V1S local normalization
    
    Each pixel in the input image is divisively normalized by the L2 norm
    of the pixels in a local neighborhood around it, and the result of this
    division is placed in the output image.   
    
    Inputs:
      hin -- a 3-dimensional array (width X height X depth)
      kshape -- kernel shape (tuple) ex: (3,3) for a 3x3 normalization 
                neighborhood
      threshold -- magnitude threshold, if the vector's length is below 
                   it doesn't get resized ex: 1.    
     
    Outputs:
      hout -- a normalized 3-dimensional array (width X height X depth)
      
    """
    
    eps = 1e-5
    kh, kw = kshape
    dtype = hin.dtype
    hsrc = hin[:].copy()

    # -- prepare hout
    hin_h, hin_w, hin_d = hin.shape
    hout_h = hin_h - kh + 1
    hout_w = hin_w - kw + 1
    hout_d = hin_d    
    hout = N.empty((hout_h, hout_w, hout_d), 'f')

    # -- compute numerator (hnum) and divisor (hdiv)
    # sum kernel
    hin_d = hin.shape[-1]
    kshape3d = list(kshape) + [hin_d]            
    ker = N.ones(kshape3d, dtype=dtype)
    size = ker.size

    # compute sum-of-square
    hsq = hsrc ** 2.
    hssq = conv(hsq, ker, 'valid').astype(dtype)

    # compute hnum and hdiv
    ys = kh / 2
    xs = kw / 2
    hout_h, hout_w, hout_d = hout.shape[-3:]
    hs = hout_h
    ws = hout_w
    hsum = conv(hsrc, ker, 'valid').astype(dtype)
    hnum = hsrc[ys:ys+hs, xs:xs+ws] - (hsum/size)
    val = (hssq - (hsum**2.)/size)
    N.putmask(val, val<0, 0) # to avoid negative sqrt
    hdiv = val ** (1./2) + eps

    # -- apply normalization
    # 'volume' threshold
    N.putmask(hdiv, hdiv < (threshold+eps), 1.)
    result = (hnum / hdiv)
    
    hout[:] = result
    return hout

# -------------------------------------------------------------------------
def v1s_filter(hin, filterbank):
    """ V1S linear filtering
    Perform separable convolutions on an image with a set of filters
    
    Inputs:
      hin -- input image (a 2-dimensional array) 
      filterbank -- list of tuples with 1d filters (row, col)
                    used to perform separable convolution
     
    Outputs:
      hout -- a 3-dimensional array with outputs of the filters 
              (width X height X n_filters)

    """

    nfilters = len(filterbank)
    shape = list(hin.shape) + [nfilters]
    hout = N.empty(shape, 'f')
    for i in xrange(nfilters):
        vectors = filterbank[i]
        row0, col0 = vectors[0]
        result = conv(conv(hin[:], row0, 'same'), col0, 'same')
        for row, col in vectors[1:]:
            result += conv(conv(hin[:], row, 'same'), col, 'same')                  
        hout[:,:,i] = result

    return hout

# -------------------------------------------------------------------------
def v1s_dimr(hin, lsum_ksize, outshape):
    """ V1S Image Downsampling
    Low-pass filter and downsample a population "image" (width X height X
    n_channels)
    
    Inputs:
      hin -- a 3-dimensional array (width X height X n_channels)
      lsum_ksize -- kernel size of the local sum ex: 17
      outshape -- fixed output shape (2d slices)
     
    Outputs:
       hout -- resulting 3-dimensional array

    """

    # -- local sum
    hin_h, hin_w, hin_d = hin.shape
    dtype = hin.dtype
    aux = N.empty(hin.shape, dtype)
    k = N.ones((lsum_ksize), 'f')
    for d in xrange(aux.shape[2]):
        aux[:,:,d] = conv(conv(hin[:,:,d], k[N.newaxis,:], 'same'), k[:,N.newaxis], 'same')
        
    # -- resample output
    hout = sresample(aux, outshape)
    return hout

# -------------------------------------------------------------------------
def sresample(src, outshape):
    """ Simple 3d array resampling

    Inputs:
      src -- a ndimensional array (dim>2)
      outshape -- fixed output shape for the first 2 dimensions
     
    Outputs:
       hout -- resulting n-dimensional array
    
    """
    
    inh, inw = inshape = src.shape[:2]
    outh, outw = outshape
    hslice = (N.arange(outh) * (inh-1.)/(outh-1.)).round().astype(int)
    wslice = (N.arange(outw) * (inw-1.)/(outw-1.)).round().astype(int)
    hout = src[hslice, :][:, wslice]    
    return hout.copy()
    
    
    
# -------------------------------------------------------------------------
def get_image(img_fname, max_edge):
    """ Return a resized image as a numpy array

    Inputs:
      img_fname -- image filename
      max_edge -- maximum edge length
     
    Outputs:
      imga -- result
    
    """
    
    # -- open image
    img = Image.open(img_fname)                
    iw, ih = img.size

    # -- resize so that the biggest edge is max_edge (keep aspect ratio)
    if iw > ih:
        new_iw = max_edge
        new_ih = int(round(1.* max_edge * ih/iw))
    else:
        new_iw = int(round(1.* max_edge * iw/ih))
        new_ih = max_edge
    img = img.resize((new_iw, new_ih), Image.BICUBIC)

    # -- convert to a numpy array
    imga = N.misc.fromimage(img)
    return imga


# -------------------------------------------------------------------------
def rephists(hin, division, nfeatures):
    """ Compute local feature histograms from a given 3d (width X height X
    n_channels) image.

    These histograms are intended to serve as easy-to-compute additional
    features that can be concatenated onto the V1-like output vector to
    increase performance with little additional complexity. These additional
    features are only used in the V1S+ (i.e. + 'easy tricks') version of
    the model. 

    Inputs:
      hin -- 3d image (width X height X n_channels)
      division -- granularity of the local histograms (e.g. 2 corresponds
                  to computing feature histograms in each quadrant)
      nfeatures -- desired number of resulting features 
     
    Outputs:
      fvector -- feature vector
    
    """

    hin_h, hin_w, hin_d = hin.shape
    nzones = hin_d * division**2
    nbins = nfeatures / nzones
    sx = (hin_w-1.)/division
    sy = (hin_h-1.)/division
    fvector = N.zeros((nfeatures), 'f')
    hists = []
    for d in xrange(hin_d):
        h = [N.histogram(hin[j*sy:(j+1)*sy,i*sx:(i+1)*sx,d], bins=nbins)[0].ravel()
             for i in xrange(division)
             for j in xrange(division)
             ]
        hists += [h]

    hists = N.array(hists, 'f').ravel()    
    fvector[:hists.size] = hists
    return fvector
