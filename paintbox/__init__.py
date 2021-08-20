# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import

import os

__all__ = ['__version__', 'test']

from astropy.tests.runner import TestRunner
test = TestRunner.make_test_runner_in(os.path.dirname(__file__))

from paintbox.sed import ParametricModel, NonParametricModel, Polynomial, \
                         CompoundSED
from paintbox.operators import LOSVDConv, Resample
from paintbox.extlaws import CCM89, C2000
from paintbox.likelihoods import StudTLogLike, StudT2LogLike, NormalLogLike, \
                                 Normal2LogLike
from paintbox.ssp_utils import CvD18, MILES
from paintbox.logspace_dispersion import disp2vel, logspace_dispersion
from paintbox.convolve import broad2lick, broad2res
from paintbox.version import version as __version__

__all__ = ["ParametricModel", "NonParametricModel", "Polynomial",
           "CompoundSED"]
__all__ += ["LOSVDConv", "Resample"]
__all__ += ["CCM89", "C2000"]
__all__ += ["StudTLogLike", "StudT2LogLike", "NormalLogLike", "Normal2LogLike"]
__all__ += ["CvD18", "MILES"]
__all__ += ["disp2vel", "logspace_dispersion"]
__all__ += ["broad2lick", "broad2res"]
