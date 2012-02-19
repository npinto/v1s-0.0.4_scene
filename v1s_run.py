#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" v1s_run module

Build and evaluate a simple V1-like model using sets of images belong to an
arbitrary number of categories (e.g. the Caltech 101 dataset)

"""

import sys, os

from v1s import V1S

EXTENSIONS = ['.png', '.jpg']

# -----------------------------------------------------------------------------
def main(param_fname, img_path):

    # -- get parameters
    param_path = os.path.abspath(param_fname)
    print "Parameters file:", param_path
    v1s_params = {}
    execfile(param_path, {}, v1s_params)

    # -- get image filenames
    img_path = os.path.abspath(img_path)
    print "Image source:", img_path
    
    # navigate tree structure and collect a list of files to process
    if not os.path.isdir(img_path):
        raise ValueError, "%s is not a directory" % (img_path)
    tree = os.walk(img_path)
    filelist = []
    categories = tree.next()[1]    
    for root, dirs, files in tree:
        if dirs != []:
            msgs = ["invalid image tree structure:"]
            for d in dirs:
                msgs += ["  "+"/".join([root, d])]
            msg = "\n".join(msgs)
            raise Exception, msg
        filelist += [ root+'/'+f for f in files if os.path.splitext(f)[-1] in EXTENSIONS ]
    filelist.sort()    
    print len(categories), "categories found:"
    print categories
    
    
    # -- create a V1S (model) object with the chosen parameters and filelist
    kwargs = v1s_params['protocol']
    kwargs['filelist'] = filelist
    v1s = V1S(**kwargs)
    
    # -- assessment of model performance
    v1s.get_performance(v1s_params['model'], v1s_params['pca_threshold'])

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        mod_fname = sys.argv[1]    
        img_path = sys.argv[2]
    except IndexError:
        progname = sys.argv[0]
        print "Usage: %s <parameter_file> <path_to_images>" % progname
        print "Example:"
        print "  %s params_simple_plus.py /tmp/101_ObjectCategories" % progname
        sys.exit(1)

    main(mod_fname, img_path)

        

