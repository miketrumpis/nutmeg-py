.. _installation:

====================
Download and Install
====================

This page lists the software that Nutmeg-Py is built upon.

Dependencies
------------

Non-Python Requirements
```````````````````````
  `Qt4 <http://qt.nokia.com/>`_
    The UI widget library used for much of Nutmeg-Py's visualization

  `gcc <http://gcc.gnu.org/>`_
    Nutmeg-Py currently builds a threaded wrapper for the FFTW3 complex DFT 
    routines, and may take advantage of other C-extension code in the future. 
    Therefore, you must have a compiler to build from
    source.  XCode (OSX) and MinGW (Windows) both include gcc.  
    Furthermore, MinGW is included in the Enthought Python Distribution for Windows.

Python Level Packages
`````````````````````
  Python_ 2.4 or later
  
  NumPy_ 1.2 or later
    Numpy equips Python with N-dimensional data/numerical data objects

  SciPy_ 0.7 or later
    Scipy is a collection of high-level, optimized scientific computing libraries, many of which are Python wrappers for widely used numerical libraries originally written in C and Fortran

  `ETS <https://svn.enthought.com/enthought>`_
    Enthought Tool Suite is a collection of tools intended for scientific computing. Nutmeg-py uses Traits, TraitsUI, and Mayavi/TVTK.

  NIPY_ (Neuroimaging in Python)
    Nutmeg-Py uses many image data types and spatial mapping tools from the
    NIPY project codebase.

  XIPY_ (Cross-modality Imaging in Python)
    Nutmeg-Py relies on XIPY for visualization. (XIPY `installation <http://miketrumpis.github.com/xipy/users/installation.html>`_ instructions).

  Matplotlib_
    2D python plotting library.

  `PyQt4 <http://www.riverbankcomputing.co.uk/software/pyqt/intro>`_
    Python bindings for the C++ Qt4 libraries.
    

Strong Recommendations
``````````````````````

  iPython_
    Interactive python environment.

  `virtualenv <http://pypi.python.org/pypi/virtualenv>`_
    Utility for changing your shell environment to a custom setup within a session

.. _building_mac:

Building from Mac OS X >= 10.4
------------------------------

With the exception of `FFTW <http://www.fftw.org/>`_, the install and run-time requirements of XIPY_ are a superset of those for Nutmeg-Py. **Please see first the installation** `instructions <http://miketrumpis.github.com/xipy/users/installation.html#building-from-mac-os-x-10-4>`_ **for XIPY!**

  **FFTW3**

  *Threaded FFTs may become a deprecated feature, and FFTW is a complicated requirement to satisfy.. no instructions yet*

Finally check out the Nutmeg-Py code, and set it up::

  git clone git://github.com/miketrumpis/nutmeg-py.git
  cd nutmeg-py && python setup.py install

.. include:: ../links.txt
