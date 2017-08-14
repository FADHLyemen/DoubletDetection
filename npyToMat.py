#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io as scio
import os
import glob


def npyToMat(dirPath=""):
    """Produces a .mat file containing all .npy in a directory.

    MATLAB file placed in same directory.

    Args:
        dirPath (String, optional): Path to directory. Absolute or relative.
    """
    if dirPath and dirPath[-1] != os.sep:
        dirPath += dirPath + os.sep

    pathNames = glob.glob(dirPath + "*.npy")
    varNames = list(map(lambda x: x[-31:-4], pathNames))
    varNames = list(map(lambda x: x.lstrip('0123456789_-'), varNames))

    scio.savemat('doubletHistogram_results.mat', dict(zip(varNames, pathNames)))


if __name__ == '__main__':
    npyToMat()
