# -*- coding: utf-8 -*-
""" 
Log-likelihood functions used in model fitting.

"""
from __future__ import print_function, division

import numpy as np
from scipy.special import gamma, digamma


__all__ = ["StudTLogLike", "StudT2LogLike", "NormalLogLike", "Normal2LogLike"]

class LogLike:
    """
    Parameters
    ----------
    observed: numpy.ndarray
        Observed spectro-photometric SED of object.
    model: paintbox SED model
        SED model used in the modelling.
    obserr: numpy.ndarray, optional
        Uncertainties in the observed SED fitting to be used in the
        weighting of the log-likelihood.
    mask: numpy.ndarray, optional
        Mask for observed data, with zeros (ones) indicating non-masked (
        masked) wavelengths.

    Attributes
    ----------
    parnames: list
        List with name of variables used in the evaluation of the
        log-likelihood.
    """

    def __init__(self, observed, model, obserr=None, mask=None):
        self.observed = observed
        self.model = model
        self.wave = self.model.wave
        self.obserr = np.ones_like(self.observed) if obserr is None else obserr
        self.mask = mask if mask is not None else np.zeros(self._N)
        self._bmask = np.where(self.mask == 0, True, False)
        self._N = len(self.wave[self._bmask])
        self._pmask = np.where(self._bmask, 1, np.nan) # Plot mask
        self.parnames = self.model.parnames.copy()
        self._nparams = len(self.parnames)

    def __add__(o1, o2):
        """ Addition of SED components. """
        return JointLogLike(o1, o2)

class NormalLogLike(LogLike):
    r""" Normal loglikelihood for SED modeling.

    The normal log-likelihood is given by

    .. math::
       :nowrap:

       \begin{equation}
          \ln \mathcal{L}(y, \sigma|\theta)= -\frac{N}{2}\ln (2\pi)
          -\frac{1}{2}\sum_{i=1}^N \left (\frac{f(\theta)- y_i}{\sigma_i}
          \right )^2 - \frac{1}{2}\sum_{i=1}^{N}\ln \sigma_i^2
       \end{equation}

    where :math:`y` is the observed spectrum, :math:`\sigma` are the
    uncertainties, :math:`\theta` is the input vector of parameters and
    and :math:`f(\theta)` is the SED model.

    """
    __doc__ = __doc__ + LogLike.__doc__

    def __init__(self, observed, model, obserr=None, mask=None):
        super().__init__(observed, model, obserr=obserr, mask=mask)

    def __call__(self, theta):
        """ Calculation of the log-likelihood. """
        e_i = (self.model(theta) - self.observed)[self._bmask]
        yerr = self.obserr[self._bmask]
        LLF = - 0.5 * self._N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.power(e_i / yerr, 2)) \
              - 0.5 * np.sum(np.log(yerr ** 2))
        return float(LLF)

    def gradient(self, theta):
        """ Gradient of the log-likelihood. """
        e_i = (self.model(theta) - self.observed)[self._bmask]
        yerr = self.obserr[self._bmask]
        g = self.model.gradient(theta)[:, self._bmask]
        grad = - np.sum((e_i / np.power(yerr, 2.))[np.newaxis, :] * g,
                        axis=1)
        return grad

class Normal2LogLike(LogLike):
    r""" Variation of the normal log-likelihood with scaled errors.

    Uncertainties in the input spectrum may be under/ over estimated in some
    occassions, leading to under/over-estimated uncertainties in parameter
    estimation. This log-likelihood includes an extra parameter to
    scale the observed uncertainties by a multiplicative factor to increase
    the likelihood of the modeling. In this case, the log-likelihood is
    given by

    .. math::
        :nowrap:

        \begin{equation}
          \ln \mathcal{L}(y, \sigma|\theta, \eta)= -\frac{N}{2}\ln (2\pi)
          -\frac{1}{2}\sum_{i=1}^N \left (\frac{f(\theta)- y_i}{\eta \sigma_i}
          \right )^2 - \frac{1}{2}\sum_{i=1}^{N}\ln \eta^2\sigma_i^2
       \end{equation}

    where :math:`y` is the observed spectrum, :math:`\sigma` are the
    uncertainties, :math:`\theta` is the input vector of parameters and
    and :math:`f(\theta)` is the SED model. The multiplicative factor
    :math:`\eta` is appended to the parnames list.


    """
    __doc__ = __doc__ + LogLike.__doc__

    def __init__(self, observed, model, obserr=None, mask=None):
        super().__init__(observed, model, obserr=obserr, mask=mask)
        self.parnames += ["eta"]
        self._nparams += 1

    def __call__(self, theta):
        model = self.model(theta[:-1])
        if np.all(model) == 0:
            return -np.infty
        e_i = (model - self.observed)[self._bmask]
        S = theta[-1]
        LLF = - 0.5 * self._N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.power(e_i / (S * self.obserr[self._bmask]), 2)) \
              - 0.5 * np.sum(np.log((S * self.obserr[self._bmask]) ** 2))
        return float(LLF)

    def gradient(self, theta):
        e_i = (self.model(theta[:-1]) - self.observed)[self._bmask]
        S = theta[-1]
        A = (e_i / np.power(S * self.obserr, 2.))[self._bmask]
        B = self.model.gradient(theta[:-1])[self._bmask]
        C = -np.sum(A[np.newaxis,:] * B, axis=1)[self._bmask]
        grad = np.zeros(len(theta))
        grad[:-1] = C
        grad[-1] = - self._N / S + \
                   np.power(S, -3) * np.sum(np.power(e_i / self.obserr, 2))
        return grad

class StudTLogLike(LogLike):
    r"""Student's t-distribution log-likelihood.

    The Student's t-distribution log-likelihood allows for robust inference of
    parameters in models containing outliers. The log-likelihood is given by

    .. math::
        :nowrap:

        \begin{equation}
            \ln p(y, \sigma | \theta, \nu ) =
            N\log \left [ \frac{\Gamma\left (\frac{\nu + 1}{2}\right )}{\sqrt{
            \pi (\nu-2)}\Gamma\left (\frac{\nu}{2} \right )}\right ]
            -\frac{1}{2}\sum_{i=1}^{N}\log \sigma_{i}^2
            -\frac{\nu+1}{2}\sum_{i=1}^N \log \left [ 1 + \frac{\left (
            y_i - f(\theta)\right )^2}{\sigma_{i}^2 (\nu-2)} \right ]
        \end{equation}

    where :math:`y` is the observed spectrum, :math:`\sigma` are the
    uncertainties, :math:`\theta` is the input vector of parameters,
    :math:`f(\theta)` is the SED model, and :math:`\nu` is the
    degree-of-freedom parameter that controls the wings of the distribution,
    which is appended to the input parnames list.
    """
    __doc__ = __doc__ + LogLike.__doc__

    def __init__(self, observed, model, obserr=None, mask=None):
        super().__init__(observed, model, obserr=obserr, mask=mask)
        self.parnames += ["nu"]
        self._nparams += 1

    def __call__(self, theta):
        nu = theta[-1]
        e_i = self.model(theta[:-1])[self._bmask] - self.observed[self._bmask]
        x = 1. + np.power(e_i / self.obserr[self._bmask], 2.) / (nu - 2)
        LLF = self._N * np.log(gamma(0.5 * (nu + 1)) /
                               np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu)) \
              - 0.5 * (nu + 1) * np.sum(np.log(x)) \
              - 0.5 * np.sum(np.log(self.obserr ** 2)) # Constant
        return float(LLF)

    def gradient(self, theta):
        grad = np.zeros(self.model._nparams + 1)
        nu = theta[-1]
        # d loglike / d theta
        e_i = self.model(theta[:-1])[self._bmask] - self.observed[self._bmask]
        x = np.power(e_i / self.obserr[self._bmask], 2.) / (nu - 2.)
        term1 = 1 / (1 + x)
        term2 = 2 * e_i / (self.obserr[self._bmask] ** 2) / (nu - 2)
        term12 = term1 * term2
        sspgrad = self.model.gradient(theta[:-1])[:, self._bmask]
        grad[:-1] = -0.5 * (nu + 1) * np.sum(term12[np.newaxis, :] *
                                             sspgrad, axis=1)
        # d loglike / d nu
        nuterm1 = 0.5 * self._N * digamma(0.5 * (nu + 1))
        nuterm2 = - 0.5 * self._N / (nu - 2)
        nuterm3 = -0.5 * self._N * digamma(0.5 * nu)
        nuterm4 = -0.5 * np.sum(np.log(1 + x))
        nuterm5 = 0.5 * (nu + 1) * np.power(nu - 2, -2) * \
                  np.sum(np.power(e_i / self.obserr, 2) * term1)
        grad[-1] = nuterm1 + nuterm2 + nuterm3 + nuterm4 + nuterm5
        return grad

class StudT2LogLike(LogLike):
    r"""Student's t-distribution log-likelihood with scaled uncertainties.

    Similar to the Normal2LogLike, this class extends the log-likelihood of
    the Student's t-distribution to include a term to scale the
    uncertainties to increase the likelihood as a way to compensate for
    under-over estimation of the observed uncertainties. In this case,
    the log-likelihood is given by

    .. math::
        :nowrap:

        \begin{equation}
            \ln p(y, \sigma | \theta, \eta, \nu ) =
            N\log \left [ \frac{\Gamma\left (\frac{\nu + 1}{2}\right )}{\sqrt{
            \pi (\nu-2)}\Gamma\left (\frac{\nu}{2} \right )}\right ]
            -\frac{1}{2}\sum_{i=1}^{N}\log \eta^2\sigma_{i}^2
            -\frac{\nu+1}{2}\sum_{i=1}^N \log \left [ 1 + \frac{\left (
            y_i - f(\theta)\right )^2}{\eta^2\sigma_{i}^2 (\nu-2)} \right ]
        \end{equation}

    where :math:`y` is the observed spectrum, :math:`\sigma` are the
    uncertainties, :math:`\theta` is the input vector of parameters,
    :math:`f(\theta)` is the SED model, :math:`\eta` is the parameter used
    to modify the scale of the uncertainties, and :math:`\nu` is the
    degree-of-freedom parameter that controls the wings of the distribution.
    Both :math:`\eta` and :math:`\nu` are appended to to input parameter list.
    """
    __doc__ = __doc__ + LogLike.__doc__

    def __init__(self, observed, model, obserr=None, mask=None):
        super().__init__(observed, model, obserr=obserr, mask=mask)
        self.parnames += ["eta", "nu"]
        self._nparams += 2

    def __call__(self, theta):
        S, nu = theta[-2:]
        e_i = self.model(theta[:-2])[self._bmask] - self.observed[self._bmask]
        x = 1. + np.power(e_i / S / self.obserr[self._bmask], 2.) / (nu - 2)
        LLF = self._N * np.log(gamma(0.5 * (nu + 1)) /
                               np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu)) \
              - 0.5 * (nu + 1) * np.sum(np.log(x)) \
              - 0.5 * np.sum(np.log((S * self.obserr[self._bmask]) ** 2))
        return float(LLF)

    def gradient(self, theta):
        raise NotImplementedError("Gradients not supported for Studt2 "
                                  "loglikelyhood.")

class JointLogLike():
    def __init__(self, logp1, logp2):
        self.logp1 = logp1
        self.logp2 = logp2
        self.parnames = list(dict.fromkeys(logp1.parnames + logp2.parnames))
        self._idxs = []
        for parlist in [logp1.parnames, logp2.parnames]:
            idxs = []
            for p in parlist:
                idxs.append(self.parnames.index(p))
            self._idxs.append(np.array(idxs))

    def __call__(self, theta):
        t1 = theta[self._idxs[0]]
        t2 = theta[self._idxs[1]]
        return self.logp1(t1) + self.logp2(t2)

    def __add__(self, other):
        """ Addition of SED components. """
        return JointLogLike(self, other)