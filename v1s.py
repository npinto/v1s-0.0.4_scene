#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time
from itertools import izip, count

from scipy import *


from PyML import *

import PyML
if PyML.__version__ >= "0.7.0":
    from PyML.classifiers import multi

from v1s_math import *
from v1s_funcs import *

# -----------------------------------------------------------------------------
class V1S(object):

    """
    V1S is a class that implements a very simple object recognition system.
    
    It combines a basic model of the human primary visual cortex
    (V1 simple cells) and a multi-class linear SVM classifier.  
    
    This class also manages all of the details of cross-validation, performing
    multiple trials, image file management, etc..

    To use it:

    1. First you need to initialize it with
       - a list of image filenames with a tree structure reflecting categories
       (ex: ['/pathto/cat1/1.png', '/pathto/cat1/2.png', '/pathto/cat2/1.png'])
       - a testing protocol (cf. __init__)
        
    2. Then you can perform a test by calling get_performance() with your
       parameters

    """


    # -------------------------------------------------------------------------
    def __init__(self, **kwargs):
        """ Constructor

        kwargs -- keywords arguments / attributes of the object, must contain:
          filelist: list of image files
          ntrain: number of training examples
          ntest: number of testing examples
          ntrials: number of trials (random draw)
          seed: seed for random number generation
          
        """

        self.__dict__.update(kwargs)

        self._cache = {}
        self._labels = None
        self._filt_l = None

    # -------------------------------------------------------------------------
    def get_performance(self, model, pca_threshold):
        """ Perform a complete, cross-validated assessment of model
            performance.

        Inputs:
          model -- list with model parameters
                   each element of the list contains a tuple of 2 elements:
                   . representation parameters (dict)
                   . feature selection options (dict)                   
          pca_threshold -- explained variance during pca projection ([0,1])
          
        Outputs:
          scores -- list of scores from each trial

        """

        # -- perform all trials
        ntrials = self.ntrials        
        scores = []        
        for n in xrange(ntrials):
            
            # run one cross-validation trial
            r = self._run_one_trial(model, pca_threshold)

            # get raw scores
            score = r.getBalancedSuccessRate()
            scores += [score]
            sa = array(scores)*100.
            
            # compute individual scores
            ind_scores = []
            cm = array(r.getConfusionMatrix())            
            catscores = []
            for i in xrange(len(cm[:])):
                good = cm[i,i]
                tot = cm[:,i].sum()
                catscore = 1.*good/tot
                catscores += [catscore]
            catscores = array(catscores)
            sel = catscores.argsort()
            i = 0
            for s in sel:
                label = self._labels[s]
                ind_score = catscores[s]*100
                ind_scores += [ " %3d - %6.2f %s \n" % \
                                ((i+1), ind_score, label) ]
                i += 1                        

            # print results
            print '=' * 80
            print "Current trial:", n+1
            print '-' * 80
            print "Current score: %.2f" % (score*100.)
            print "Current individual scores:"
            print "".join(ind_scores),
            print '-' * 80
            print "All scores:"
            print "\n".join([ " (%d) %6.2f"%(i+1,s*100)
                              for i,s in izip(count(),scores)])
            print '-' * 80
            mstd = sa.std()
            mstderr = mstd/sqrt(sa.size)
            print n+1, "Average: %.2f (std=%.2f, stderr=%.2f)" % \
                  (sa.mean(), mstd, mstderr)
            print '=' * 80
                        
        return scores

    # -------------------------------------------------------------------------
    def _run_one_trial(self, model, pca_threshold):
        """ Perform one trial (i.e. one train/test split) of model performance 
        assessment
        
        The following steps are performed here:
        
        1) Images are shuffled into random train and test sets
        2) The V1S model is applied to the training images and feature vectors 
            are built from its outputs
        3) The training set is sphered, and a PCA projection is computed
        4) The V1S model is applied to the test images, and the resulting
            feature vectors are sphered and PC projected using parameters
            obtained from the training data

        Inputs:
          model -- list with model parameters (cf. self.get_performance)
          pca_threshold -- explained variance during pca projection ([0,1])
          
        Outputs:
          results -- performance achieved (PyML.assess.Results object)
        
        """
        
        filelist = self.filelist
        ntrain = self.ntrain
        ntest = self.ntest        

        # -- Organize images into the appropriate categories
        cats = {}
        for f in filelist:
            cat = "/".join(f.split('/')[:-1])
            name = f.split('/')[-1]
            if cat not in cats:
                cats[cat] = [name]
            else:
                cats[cat] += [name]

        # -- Shuffle the images into a new random order
        filelists_dict = {}
        for cat in cats:
            filelist = cats[cat]
            random.seed(self.seed)
            random.shuffle(filelist)
            self.seed += 1
            filelist = [ cat + '/' + f for f in filelist ]
            filelists_dict[cat] = filelist

        # -- Get training examples
        #   (i.e. read in the images, apply the model, and construct a 
        #    feature vector)
        #   The first ntrain examples (of the shuffled list) will be used for
        #   training.
        print "="*80
        print "Training..."
        print "="*80

        train_fvectors, train_flabels, train_fnames = \
                        self._get_fvectors_flabels_fnames(filelists_dict,
                                                          model, 0, ntrain)
        print "training samples:", len(train_fnames)

        # -- Sphere the training data
        #    (we will later use the sphering parameters obtained here to
        #     "sphere" the test data)
        print "sphering data..."
        v_sub = train_fvectors.mean(axis=0)
        train_fvectors -= v_sub
        v_div = train_fvectors.std(axis=0)
        putmask(v_div, v_div==0, 1)
        train_fvectors /= v_div
        
        # -- Reduce dimensionality using a pca / eigen subspace projection
        nvectors, vsize = train_fvectors.shape        
        if nvectors < vsize:
            print "pca...", 
            print train_fvectors.shape, "=>", 
            U,S,V = fastsvd(train_fvectors)
            eigvectors = V.T
            i = tot = 0
            S **= 2.
            while (tot <= pca_threshold) and (i < S.size):
                tot += S[i]/S.sum()
                i += 1
            eigvectors = eigvectors[:, :i+1]
            train_fvectors = dot(train_fvectors, eigvectors)
            print train_fvectors.shape

        # -- Train svm    
        print "creating dataset..."
        if PyML.__version__ < "0.7.0":
            train_data = datafunc.VectorDataSet(X=train_fvectors.astype(double),
                                                L=train_flabels)
        else:
            train_data = containers.VectorDataSet(train_fvectors.astype(double),
                                                  L=train_flabels)
        print "creating classifier..."
        if len(filelists_dict) > 2:
            clas = multi.OneAgainstRest(svm.SVM())
        else:
            clas = svm.SVM()
        print "training svm..."
        clas.train(train_data)

        # -- Get testing examples
        #    (read files, apply model, sphere using parameters from training
        #    data, project onto eigensubspace obtained from training data)
        print "="*80
        print "Testing..."
        print "="*80

        # -- A "hook" function to pass in that performs the 
        #    sphering/projection according to parameters obtained from the
        #    training data
        if nvectors < vsize:
            hook = lambda vector: dot(((vector-v_sub) / v_div), eigvectors)
        else:
            hook = lambda vector: ((vector-v_sub) / v_div)

        test_fvectors, test_flabels, test_fnames = \
                       self._get_fvectors_flabels_fnames(filelists_dict, model,
                                                         ntrain, ntrain+ntest,
                                                         hook)
        print "testing samples:", len(test_fnames)
        
        # a safety check that there are no duplicates across the test and 
        # train sets
        for test in test_fnames:
            if test in train_fnames:
                raise ValueError, "image already included in train set"

        # -- Test trained svm    
        print "creating dataset..."
        if PyML.__version__ < "0.7.0":
            test_data = datafunc.VectorDataSet(X=test_fvectors.astype(double),
                                               L=test_flabels)
        else:
            test_data = containers.VectorDataSet(test_fvectors.astype(double),
                                                 L=test_flabels)
        self._labels = clas.labels.classLabels
        
        print "testing..."
        results = clas.test(test_data)
        results.computeStats()

        print results
        return results

    # -------------------------------------------------------------------------
    def _get_fvectors_flabels_fnames(self, filelists_dict, model, idx_start,
                                     idx_end, hook=None):
        """ Read in data from a specific set of image files, apply the model,
            and generate feature vectors for each image.

        Inputs:
          filelists_dict -- dict of image files where:
                           . each key is a category,
                           . each corresponding value is a list of images
          model -- list with model parameters (cf. self.get_performance)
          idx_start -- start index in image list
          idx_end -- end index in image list
          hook -- hook function, if not None this function will take a vector
                  perform needed computation and return the resulting vector

        Outputs:
          fvectors -- array with feature vectors          
          flabels -- list with corresponding class labels
          fnames -- list with corresponding file names        

        """
        
        flabels = []
        fnames = []
        catlist = filelists_dict.keys()
        catlist.sort()

        # -- get fvectors size and initialize it
        print "Initializing..."
        f = filelists_dict[catlist[0]][0]
        if f in self._cache:
            vector = self._cache[f]
        else:
            vector = self._generate_repr(f, model)
            self._cache[f] = vector
            
        if hook is not None:
            vector = hook(vector)        
        nvectors = 0
        for cat in catlist:
            nvectors += len(filelists_dict[cat][idx_start:idx_end])

        fvectors = empty((nvectors, vector.size), 'f')

        # -- get all vectors
        i = 0
        for cat in catlist:
            print "get data from class:", cat

            for f in filelists_dict[cat][idx_start:idx_end]:
                txt = os.path.split(f)[-1]
                print "  file:", txt,
                
                if f in self._cache:
                    vector = self._cache[f]
                else:
                    vector = self._generate_repr(f, model)
                    self._cache[f] = vector
                    
                if hook is not None:
                    vector = hook(vector)        

                print "vsize:", vector.size

                fvectors[i] = vector
                flabels += [os.path.split(cat)[-1]]
                fnames += [f]
                i += 1
                
        return fvectors, flabels, fnames

    # -------------------------------------------------------------------------
    def _generate_repr(self, img_fname, model):
        """ Apply the simple V1-like model and rearrange the outputs into
            a feature vector suitable for further processing (e.g. 
            dimensionality reduction & classification). Model parameters
            determine both how the V1-like representation is built and which
            additional synthetic features (if any) are included in the 
            final feature vector.
            
            Most of the work done by this function is handled by the helper
            function _part_generate_repr.

        Inputs:
          img_fname -- image filename
          model -- list with model parameters (cf. self.get_performance)

        Outputs:
          fvector -- corresponding feature vector
          
        """
        
        #return randn(100)

        all = []
        for params, featsel in model:
            r = self._part_generate_repr(img_fname, params, featsel)
            all += [r]
            
        fvector = concatenate(all)

        return fvector

    # -------------------------------------------------------------------------
    def _part_generate_repr(self, img_fname, params, featsel):
        """ Applies a simple V1-like model and generates a feature vector from
        its outputs. See description of _generate_repr, above.

        Inputs:
          img_fname -- image filename
          params -- representation parameters (dict)
          featsel -- features to include to the vector (dict)

        Outputs:
          fvector -- corresponding feature vector                  

        """
        
        # -- get image as an array
        orig_imga = get_image(img_fname, params['preproc']['max_edge'])

        # -- 0. preprocessing
        imga0 = orig_imga.astype('f') / 255.0        
        if imga0.ndim == 3:
            # grayscale conversion
            imga0 = 0.2989*imga0[:,:,0] + \
                    0.5870*imga0[:,:,1] + \
                    0.1140*imga0[:,:,2]
            
        # smoothing
        lsum_ksize = params['preproc']['lsum_ksize']
        mode = 'same'
        if lsum_ksize is not None:
            k = ones((lsum_ksize), 'f') / lsum_ksize
            imga0 = conv(conv(imga0, k[newaxis,:], mode), k[:,newaxis], mode)
            imga0 -= imga0.mean()
            if imga0.std() != 0:
                imga0 /= imga0.std()
        
        # -- 1. input normalization
        imga1 = v1s_norm(imga0[:,:,newaxis], **params['normin'])
        
        # -- 2. linear filtering
        filt_l = self._get_gabor_filters(params['filter'])
        imga2 = v1s_filter(imga1[:,:,0], filt_l)

        # -- 3. simple non-linear activation (clamping)
        minout = params['activ']['minout'] # sustain activity
        maxout = params['activ']['maxout'] # saturation
        imga3 = imga2.clip(minout, maxout)

        # -- 4. output normalization
        imga4 = v1s_norm(imga3, **params['normout'])

        # -- 5. volume dimension reduction
        imga5 = v1s_dimr(imga4, **params['dimr'])
        output = imga5
        
        # -- 6. handle features to include
        feat_l = []
        
        # include representation output ?
        f_output = featsel['output']
        if f_output:
            feat_l += [output.ravel()]

        # include grayscale values ?
        f_input_gray = featsel['input_gray']
        if f_input_gray is not None:
            shape = f_input_gray
            feat_l += [misc.imresize(imga0, shape).ravel()]
        
        # include color histograms ?
        f_input_colorhists = featsel['input_colorhists']
        if f_input_colorhists is not None:
            nbins = f_input_colorhists
            colorhists = empty((3,nbins), 'f')
            if orig_imga.ndim == 3:
                for d in xrange(3):
                    h = histogram(orig_imga[:,:,d].ravel(),
                                  bins=nbins,
                                  range=[0,255])
                    binvals = h[0].astype('f')
                    colorhists[d] = binvals
            else:
                h = histogram(orig_imga[:,:].ravel(),
                              bins=nbins,
                              range=[0,255])
                binvals = h[0].astype('f')
                colorhists[:] = binvals
                
            feat_l += [colorhists.ravel()]
        
        # include input norm histograms ? 
        f_normin_hists = featsel['normin_hists']
        if f_normin_hists is not None:
            division, nfeatures = f_normin_hists
            feat_l += [rephists(imga1, division, nfeatures)]
        
        # include filter output histograms ? 
        f_filter_hists = featsel['filter_hists']
        if f_filter_hists is not None:
            division, nfeatures = f_filter_hists
            feat_l += [rephists(imga2, division, nfeatures)]
        
        # include activation output histograms ?     
        f_activ_hists = featsel['activ_hists']
        if f_activ_hists is not None:
            division, nfeatures = f_activ_hists
            feat_l += [rephists(imga3, division, nfeatures)]
        
        # include output norm histograms ?     
        f_normout_hists = featsel['normout_hists']
        if f_normout_hists is not None:
            division, nfeatures = f_normout_hists
            feat_l += [rephists(imga4, division, nfeatures)]
        
        # include representation output histograms ? 
        f_dimr_hists = featsel['dimr_hists']
        if f_dimr_hists is not None:
            division, nfeatures = f_dimr_hists
            feat_l += [rephists(imga5, division, nfeatures)]

        # -- done !
        fvector = concatenate(feat_l)
        return fvector

    # -------------------------------------------------------------------------
    def _get_gabor_filters(self, params):
        """ Return a Gabor filterbank (generate it if needed)
            
        Inputs:
          params -- filters parameters (dict)

        Outputs:
          filt_l -- filterbank (list)

        """

        if self._filt_l is None:        
            # -- get parameters
            fh, fw = params['kshape']
            orients = params['orients']
            freqs = params['freqs']
            phases = params['phases']
            nf =  len(orients) * len(freqs) * len(phases)
            fbshape = nf, fh, fw
            gsw = fw/5.
            gsh = fw/5.
            xc = fw/2
            yc = fh/2
            filt_l = []

            # -- build the filterbank
            for freq in freqs:
                for orient in orients:
                    for phase in phases:
                        # create 2d gabor
                        filt = gabor2d(gsw,gsh,
                                       xc,yc,
                                       freq,orient,phase,
                                       (fw,fh))

                        # vectors for separable convolution
                        U,S,V = fastsvd(filt)
                        tot = 0
                        vectors = []
                        idx = 0
                        S **= 2.
                        while tot <= params['sep_threshold']:
                            row = (U[:,idx]*S[idx])[:, newaxis]
                            col = (V[idx,:])[newaxis, :]
                            vectors += [(row,col)]
                            tot += S[idx]/S.sum()
                            idx += 1                             

                        filt_l += [vectors]

            self._filt_l = filt_l
            
        return self._filt_l
       
