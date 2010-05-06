#!/usr/bin/env python
import sys
from glob import glob
from distutils.cmd import Command
# we use cython to compile the module if we have it
try:
    import Cython
except ImportError:
    has_cython = False
else:
    has_cython = True
import numpy as np

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.get_version('nutmeg/version.py')
    config.add_subpackage('nutmeg', 'nutmeg')

    return config

################################################################################
# For some commands, use setuptools

if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb', 
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    from setup_egg import extra_setuptools_args

# extra_setuptools_args can be defined from the line above, but it can
# also be defined here because setup.py has been exec'ed from
# setup_egg.py.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()

# Construct the Cython extension
from build_helpers import make_cython_ext
sutils_ext, cmdclass = make_cython_ext(
    'nutmeg.stats._sutils',
    has_cython,
    include_dirs = [np.get_include()]
    )

from numpy.distutils.command.build_ext import build_ext
cmdclass.update( dict(build_ext=build_ext) )

def main(**extra_args):
    from numpy.distutils.core import setup
    setup(name = 'nutmeg-py',
          description='Magnetoencephalography imaging package in Python',
          author = 'S Dalal, J Zumer, M Trumpis',
          author_email = 'mtrumpis@gmail.com',
          url = 'http://nutmeg.berkeley.edu',
          long_description = '',
          configuration=configuration,
          cmdclass=cmdclass,
          ext_modules=[sutils_ext],
          scripts=glob('scripts/*.py'),
          **extra_args)

if __name__ == '__main__':
    main(**extra_setuptools_args)
