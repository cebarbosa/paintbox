# -*- coding: utf-8 -*-
""" 

Created on 31/08/18

Author : Carlos Eduardo Barbosa


"""

from __future__ import division, absolute_import, print_function

import os

import matplotlib.pyplot as plt

# Location of the project and data
home = "/home/kadu/Dropbox/bsf/blind_test"
data_dir = os.path.join(home, "data")

# Settings for the plots
plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = 'Computer Modern'


