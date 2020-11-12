Building models with **paintbox**
---------------------------------

In **paintbox**, the models used to describe the observed spectrum
and/or spectral energy distribution of a galaxy are build from a
combination of spectral components, including the stellar continuum,
emission lines for the gas, etc. Moreover, the parametrization of the
model, i.e., the specific details about how these spectral elements are
combined, are defined interactively. Below, we illustrate how to
generate these spectral components in practice and how to combine them
to make a model.

Using CvD models
~~~~~~~~~~~~~~~~

Here we use the CvD single stellar popupation models, and the response
functions for several elements, prepared in this
`tutorial <https://paintbox.readthedocs.io/en/latest/preparing_models.html#preparing-cvd-models>`__
to illustrate how to create the basic elements for the stellar
continuum.

.. code:: ipython3

    import os
    
    from astropy.io import fits
    from astropy.table import Table
    
    import paintbox as pb

.. code:: ipython3

    ssp_file = "templates/VCJ17_varydoublex.fits"
    templates = fits.getdata(ssp_file, ext=0)


::


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-6-2e920fed7856> in <module>
          1 ssp_file = "templates/VCJ17_varydoublex.fits"
    ----> 2 templates = fits.getdata(ssp_file, ext=0)
    

    ~/anaconda3/envs/py37-env/lib/python3.7/site-packages/astropy/io/fits/convenience.py in getdata(filename, header, lower, upper, view, *args, **kwargs)
        187     mode, closed = _get_file_mode(filename)
        188 
    --> 189     hdulist, extidx = _getext(filename, mode, *args, **kwargs)
        190     try:
        191         hdu = hdulist[extidx]


    ~/anaconda3/envs/py37-env/lib/python3.7/site-packages/astropy/io/fits/convenience.py in _getext(filename, mode, ext, extname, extver, *args, **kwargs)
       1047         raise TypeError('extver alone cannot specify an extension.')
       1048 
    -> 1049     hdulist = fitsopen(filename, mode=mode, **kwargs)
       1050 
       1051     return hdulist, ext


    ~/anaconda3/envs/py37-env/lib/python3.7/site-packages/astropy/io/fits/hdu/hdulist.py in fitsopen(name, mode, memmap, save_backup, cache, lazy_load_hdus, **kwargs)
        163 
        164     return HDUList.fromfile(name, mode, memmap, save_backup, cache,
    --> 165                             lazy_load_hdus, **kwargs)
        166 
        167 


    ~/anaconda3/envs/py37-env/lib/python3.7/site-packages/astropy/io/fits/hdu/hdulist.py in fromfile(cls, fileobj, mode, memmap, save_backup, cache, lazy_load_hdus, **kwargs)
        403         return cls._readfrom(fileobj=fileobj, mode=mode, memmap=memmap,
        404                              save_backup=save_backup, cache=cache,
    --> 405                              lazy_load_hdus=lazy_load_hdus, **kwargs)
        406 
        407     @classmethod


    ~/anaconda3/envs/py37-env/lib/python3.7/site-packages/astropy/io/fits/hdu/hdulist.py in _readfrom(cls, fileobj, data, mode, memmap, save_backup, cache, lazy_load_hdus, **kwargs)
       1052             if not isinstance(fileobj, _File):
       1053                 # instantiate a FITS file object (ffo)
    -> 1054                 fileobj = _File(fileobj, mode=mode, memmap=memmap, cache=cache)
       1055             # The Astropy mode is determined by the _File initializer if the
       1056             # supplied mode was None


    ~/anaconda3/envs/py37-env/lib/python3.7/site-packages/astropy/utils/decorators.py in wrapper(*args, **kwargs)
        533                     warnings.warn(message, warning_type, stacklevel=2)
        534 
    --> 535             return function(*args, **kwargs)
        536 
        537         return wrapper


    ~/anaconda3/envs/py37-env/lib/python3.7/site-packages/astropy/io/fits/file.py in __init__(self, fileobj, mode, memmap, overwrite, cache)
        191             self._open_fileobj(fileobj, mode, overwrite)
        192         elif isinstance(fileobj, str):
    --> 193             self._open_filename(fileobj, mode, overwrite)
        194         else:
        195             self._open_filelike(fileobj, mode, overwrite)


    ~/anaconda3/envs/py37-env/lib/python3.7/site-packages/astropy/io/fits/file.py in _open_filename(self, filename, mode, overwrite)
        572 
        573         if not self._try_read_compressed(self.name, magic, mode, ext=ext):
    --> 574             self._file = fileobj_open(self.name, IO_FITS_MODES[mode])
        575             self.close_on_error = True
        576 


    ~/anaconda3/envs/py37-env/lib/python3.7/site-packages/astropy/io/fits/util.py in fileobj_open(filename, mode)
        394     """
        395 
    --> 396     return open(filename, mode, buffering=0)
        397 
        398 


    FileNotFoundError: [Errno 2] No such file or directory: 'templates/VCJ17_varydoublex.fits'


