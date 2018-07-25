# -*- coding: utf-8 -*-
""" 

Created on 26/04/18

Author : Carlos Eduardo Barbosa

"""

import os
import sys
import getpass

basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, basedir)

if getpass.getuser() == "kadu":
    workdir = "/home/kadu/Dropbox/bsf/"
else:
    workdir = "/scratch/5386553/projects/bsf"
plots_dir = os.path.join(workdir, "plots")
dirs = [plots_dir]
for d in dirs:
    if os.path.exists(d):
        continue
    os.mkdir(d)