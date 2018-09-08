# -*- coding: utf-8 -*-
""" 

Created on 26/04/18

Author : Carlos Eduardo Barbosa

"""

import os
import sys
import getpass

if getpass.getuser() == "kadu":
    home = "/home/kadu/Dropbox/bsf-sim"
else:
    home = "/sto/home/cebarbosa/bsf-sim"

if getpass.getuser() == "kadu":
    workdir = "/home/kadu/Dropbox/bsf/"
    basedir = os.path.dirname(os.path.abspath(__file__))
else:
    workdir = "/scratch/5386553/bsf"
    basedir = "/scratch/5386553/repos/bsf"
sys.path.insert(0, basedir)
plots_dir = os.path.join(workdir, "plots")
dirs = [plots_dir]
for d in dirs:
    if os.path.exists(d):
        continue
    os.mkdir(d)