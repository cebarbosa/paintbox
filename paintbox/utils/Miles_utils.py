import os
import glob

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.io import fits
from tqdm import tqdm
from spectres import spectres
from scipy.ndimage.filters import gaussian_filter1d
from paintbox.sed import ParametricModel, PaintboxBase

from .dispersion_constant_velscale import dispersion_const_velscale

class EMILES(PaintboxBase):

