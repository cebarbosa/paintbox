# -*- coding: utf-8 -*-
"""

Basic SED classes for handling stellar population and emission lines
based on precomputed templates and polynomials.

"""
from __future__ import print_function, division

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import legendre
import matplotlib.pyplot as plt

__all__ = ["ParametricModel", "NonParametricModel", "Polynomial",
           "CompoundSED", "PaintboxBase"]

class PaintboxBase():

    @property
    def parnames(self):
        """ List with names of the parameters of the model. """
        return self._parnames

    @parnames.setter
    def parnames(self, newnames):
        if not isinstance(newnames, list):
            raise ValueError("Parnames should be set as a list.")
        if not all([isinstance(_, str) for _ in newnames]):
            raise ValueError("Parnames should be a list of strings.")
        self._parnames = newnames

    def gradient(self, theta):
        raise NotImplementedError("Gradients not implemented for this class.")

    def __call__(self, theta):
        raise NotImplementedError("Call not implemented for this class.")

    def __add__(self, o):
        """ Addition between two SED components. """
        return CompoundSED(self, o, "+")

    def __mul__(self, o):
        """  Multiplication between two SED components. """
        return CompoundSED(self, o, "*")

    def fix(self, fixed_vals):
        """ Fix a set of parameters in parnames using a dictionary. """
        return _FixSEDPars(self, fixed_vals)

    def constrain_duplicates(self):
        """ Constrain the parameters with the same name to have the same
        value during calling. """
        return _ConstrainDuplicates(self)

    def plot(self, theta, ax=None, plottype=None, **kwargs):
        """ Plot the model for a given set of parameters. """
        ax = plt.gca() if ax is None else ax
        plottype = "plot" if plottype is None else plottype
        plotter = getattr(ax, plottype)
        plotter(self.wave, self(theta), **kwargs)

class CompoundSED(PaintboxBase):
    """ Combination of SED models.

    The CompoundSED class allows the combination of any number of SED model
    components using addition and / or multiplication, as long as the input
    classes have the same wavelength dispersion. It can be accessed directly,
    but the recommended way is to use model operators.

    Attributes
    ----------
    parnames: list
        The new parnames list is a concatenation of the input SED models.
    wave: numpy.ndarray, astropy.quantities.Quantity
        Wavelength array.
    """

    def __init__(self, o1, o2, op):
        """
        Parameters
        ----------
        o1, o2: SED model components
            Input SED models to be combined either by multiplication or
            addition.
        op: str
            Operation of the combination, either "+" or "*".
        """
        msg = "Components with different wavelenghts cannot be combined!"
        assert np.allclose(o1.wave, o2.wave), msg
        self.__op = op
        msg = "Operations allowed in combination of SED components are + and *."
        assert self.__op in ["+", "*"], msg
        self.o1 = o1
        self.o2 = o2
        self.wave = self.o1.wave
        self.parnames = self.o1.parnames + self.o2.parnames
        self._nparams = len(self.parnames)
        self._grad_shape = (self._nparams, len(self.wave))


    def __call__(self, theta):
        """ SED model for combined components at point theta. """
        theta1 = theta[:self.o1._nparams]
        theta2 = theta[self.o1._nparams:]
        if self.__op == "+":
            return self.o1(theta1) + self.o2(theta2)
        elif self.__op == "*":
            return self.o1(theta1) * self.o2(theta2)

    def gradient(self, theta):
        """ Gradient of the combined SED model at point theta. """
        n = self.o1._nparams
        theta1 = theta[:n]
        theta2 = theta[n:]
        grad = np.zeros(self._grad_shape)
        if self.__op == "+":
            grad[:n] = self.o1.gradient(theta1)
            grad[n:] = self.o2.gradient(theta2)
        elif self.__op == "*":
            grad[:n] = self.o1.gradient(theta1) * self.o2(theta2)
            grad[n:] = self.o2.gradient(theta2) * self.o1(theta1)
        return np.squeeze(grad)

class _FixSEDPars(PaintboxBase):
    def __init__(self, sed, fixed_vals):
        """ Fix parameters of SED model. """
        self.sed = sed
        self.wave = self.sed.wave
        self.fixed_vals = fixed_vals
        self._parnames = [_ for _ in sed.parnames if _ not in fixed_vals]
        self._free_idxs = {}
        self._ntot = len(self.sed.parnames)
        for param in self._parnames:
            self._free_idxs[param] = np.where( \
                                np.array(self.sed.parnames) == param)[0]
        self._fixed_idxs = {}
        for param, val in self.fixed_vals.items():
            self._fixed_idxs[param] = np.where(
                                      np.array(self.sed.parnames) == param)[0]

    def __call__(self, theta):
        """ Calculates the constrained model. """
        t = np.zeros(self._ntot)
        for param, val in zip(self.parnames, theta):
            t[self._free_idxs[param]] = val
        for param, val in self.fixed_vals.items():
            idx = self._fixed_idxs[param]
            t[idx] = val
        return self.sed(t)

class _ConstrainDuplicates(PaintboxBase):
    """ Constrain parameters of an SED model.

    The combination of SED models may result in models with repeated
    parameters at different locations of the parnames list. This class allows
    the simplification of the input model by finding and constraining all
    instances of repeated parameters to the same value.

    Attributes
    ----------
    parnames: list
        The new parnames list is a concatenation of the input SED models,
        simplified in relation to the input model.
    wave: numpy.ndarray, astropy.quantities.Quantity
        Wavelength array.

    Methods
    -------
    __call__(theta)
        Returns the SED model according the parameters given in theta.

    """
    def __init__(self, sed):
        self.sed = sed
        self._parnames = list(dict.fromkeys(sed.parnames))
        self.wave = self.sed.wave
        self._nparams = len(self._parnames)
        self._ntot = len(self.sed.parnames)
        self._idxs = {}
        for param in self._parnames:
            self._idxs[param] = np.where( \
                                np.array(self.sed.parnames) == param)[0]

    def __call__(self, theta):
        """ Calculates the constrained model. """
        t = np.zeros(self._ntot)
        for param, val in zip(self._parnames, theta):
            t[self._idxs[param]] = val
        return self.sed(t)


class ParametricModel(PaintboxBase):
    """
    Interpolation of SED model templates parametrically.

    This class allows the linear interpolation of SED templates, such as SSP
    models, based on a table of parameters and their SEDs.

    Warning: The linear interpolation is currently based on
    scipy.RegularGridInterpolator for better performance with large number of
    input models, thus the input data must be in the form of a regular grid.

    Parameters
    ----------
    wave: ndarray, astropy.units.Quantity
        Wavelenght array of the model.
    params: astropy.table.Table
        Table with parameters of the models.
    templates: 2D ndarray
        The SED templates with dimensions (len(params), len(wave))

    """
    def __init__(self, wave, params, templates):
        """ """
        self.wave = wave
        self.params = params
        self.templates = templates
        self._parnames = self.params.colnames.copy()
        self._n = len(wave)
        self._nparams = len(self.parnames)
        # Interpolating models
        x = self.params.as_array()
        pdata = x.view((x.dtype[0], len(x.dtype.names)))
        nodes = []
        for param in self.parnames:
            x = np.unique(self.params[param]).data
            nodes.append(x)
        coords = np.meshgrid(*nodes, indexing='ij')
        dim = coords[0].shape + (self._n,)
        templates = np.zeros(dim)
        with np.nditer(coords[0], flags=['multi_index']) as it:
            while not it.finished:
                multi_idx = it.multi_index
                x = np.array([coords[i][multi_idx] for i in range(len(coords))])
                idx = (pdata == x).all(axis=1).nonzero()[0]
                if idx.size > 0:
                    templates[multi_idx] = self.templates[idx]
                it.iternext()
        self._interpolator = RegularGridInterpolator(nodes, templates,
                                     bounds_error=False, fill_value=0)
        ########################################################################
        # Get grid points to handle derivatives
        inner_grid = []
        thetamin = []
        thetamax = []
        eps = []
        limits = {}
        for par in self.parnames:
            vmin = np.min(self.params[par].data)
            vmax = np.max(self.params[par].data)
            thetamin.append(vmin)
            thetamax.append(vmax)
            unique = np.unique(self.params[par].data)
            eps.append(1e-4 * np.diff(unique).min())
            inner_grid.append(unique[1:-1])
            limits[par] = (vmin, vmax)
        self._limits = limits
        self._thetamin = np.array(thetamin)
        self._thetamax = np.array(thetamax)
        self._inner_grid = inner_grid
        self._eps = np.array(eps)

    def __call__(self, theta):
        """ Call for interpolated model at a given point theta.

        Parameters
        ----------
        theta: ndarray
            Point where the model is computed, with parameters in
            the same order of parnames. Points outside of the convex hull of
            the models (as defined in the limits) are set to zero.

        Returns
        -------
        SED model at location theta.
        """
        out = self._interpolator(theta)
        return np.squeeze(out)

    def __add__(self, o):
        """ Addition between two SED components. """
        return CompoundSED(self, o, "+")

    def __mul__(self, o):
        """  Multiplication between two SED components. """
        return CompoundSED(self, o, "*")

    @property
    def limits(self):
        """ Lower and upper limits of the parameters. """
        return self._limits

    def gradient(self, theta):
        """ Gradient of models at a given point theta.

        Gradients are computed with simple finite difference. If the input
        point is among the points used for interpolation, the gradient is not
        defined, returning zero instead.

        Parameters
        ----------
        theta: ndarray
            Point where the gradient of the model is computed,
            with parameters in the same order of parnames. Points outside
            of the convex hull of the models are set to zero.
        eps: float or ndarray, optional
            Step used in the finite difference calculation. Default is 1e-6.

        """
        # Clipping theta to avoid border problems
        theta = np.atleast_2d(theta)
        dim = theta.shape[0]
        tmin = np.tile(self._thetamin + 2 * self._eps, dim).reshape(
                       theta.shape[0],-1)
        tmax = np.tile(self._thetamax - 2 * self._eps, dim).reshape(
                       theta.shape[0],-1)
        theta = np.maximum(theta, tmin)
        theta = np.minimum(theta, tmax)
        grads = np.zeros((dim, self._nparams, self._n))
        for i in range(self._nparams):
            epsilon = np.zeros_like((theta))
            eps = self._eps[i]
            epsilon[:,i] = eps
            # Check if data point is in inner grid
            tp = theta + epsilon
            tm = theta - epsilon
            g = (self.__call__(tp) - self.__call__(tm)) / (2 * eps)
            # Gradients not well-dined become zero
            isin = np.isin(theta[:, i], self._inner_grid[i])
            g[isin] = 0
            grads[:,i,:] = g
        return grads

class NonParametricModel(PaintboxBase):
    """
    Weighted linear combination of SED models.

    This class allows the combination of a set of templates based on
    different weights.

    Attributes
    ----------
    parnames: list
        Name of the templates.

    """
    def __init__(self, wave, templates, names=None):
        """ 
        Parameters
        ----------
        wave: ndarray, Quantity
            Common wavelenght array of all templates.
        templates: 2D ndarray
            SED models with dimensions (N, len(wave)), where N=number of
            templates.
        names: list
            Name of the templates. Defaults to [temp, ..., tempN]
        """
        self.wave = wave
        self.templates = np.atleast_2d(templates)
        self._nparams = len(self.templates)
        self._n = len(templates)
        names = ["temp{}".format(n+1) for n in range(self._n)] if \
                 names is None else names
        self.parnames = names
        self._nparams = len(self.parnames)

    def __call__(self, theta):
        """ Returns the dot product of a vector theta with the templates.

        Parameters
        ----------
        theta: ndarray
            Vector with weights of the templates.

        Returns
            Dot product of theta with templates.
        """
        return np.dot(theta, self.templates)

    def gradient(self, theta):
        """ Gradient of the dot product with weights theta.

        This routine returns simply the templates, but it has an argument
        theta only to keep calls consistently across different SED
        components.
        """
        return self.templates

class Polynomial(PaintboxBase):
    """
    Polynomial SED component.

    This class produces Legendre polynomials that can be either added or
    multiplied with other SED components.

    Attributes
    ----------
    wave: ndarray, Quantity
        Wavelength array of the polynomials
    degree: int
        Order of the Legendre polynomial.
    poly: 2D ndarray
        Static Legendre polynomials array used in the calculations.
    pname: str (optional)
        Name of the polynomial to be used in parnames.
    zeroth: bool (optional)
        Controls used of zeroth order polynomial. Default is True (zero
        order is used).

    """

    def __init__(self, wave, degree, pname=None, zeroth=True):
        self.wave = wave
        self.degree = degree
        self._x = 2 * ((self.wave - self.wave.min()) /
                       (self.wave.max() - self.wave.min()) - 0.5)
        self.pname = "p" if pname is None else pname
        orders = np.arange(self.degree +1)
        if not zeroth:
            orders = orders[1:]
        self.poly = np.zeros((len(orders), len(self._x)))
        for i, o in enumerate(orders):
            self.poly[i] = legendre(o)(self._x)
        self._parnames = ["{}_{}".format(self.pname, o) for o in orders]
        self._nparams = len(self.parnames)

    def __call__(self, theta):
        """ Calculation of the polynomial with weights theta. """
        return np.dot(theta, self.poly)

    def gradient(self, theta):
        """ Gradient of the polynomial with weights theta. """
        return self.poly

class NSSPs():
    """ Stellar population model. """
    def __init__(self, wave, params, templates, ncomp=2, wprefix=None):
        assert isinstance(ncomp, int), "Number of components should be an " \
                                       "integer."

        self.params = params
        self.wave = wave
        self.templates = templates
        self.ncomp = np.max([ncomp, 1])
        self.wprefix = "w" if wprefix is None else wprefix
        self.ssp = ParametricModel(self.wave, self.params, self.templates)
        self.ncols = len(self.params.colnames)
        self._nparams = self.ncomp * (len(self.params.colnames) + 1)
        self.shape = (self.nparams, len(self.wave))
        # Set parameter names
        self.parnames = []
        for n in range(self.ncomp):
            for p in [self.wprefix] + self.params.colnames:
                self.parnames.append("{}_{}".format(p, n+1))

    def __call__(self, theta):
        p = theta.reshape(self.ncomp, -1)
        return np.dot(p[:,0], self.ssp(p[:, 1:]))

    def gradient(self, theta):
        grad = np.zeros(self.shape)
        ps = theta.reshape(self.ncomp, -1)
        ssps = self.ssp(ps[:,1:])
        for i, p in enumerate(ps):
            idx = i * (self.ncols + 1)
            # dF/dFi
            grad[idx] = ssps[i]
            # dF / dSSPi
            grad[idx+1:idx+1+self.ncols] = p[0] * self.ssp.gradient(p[1:])
        return grad