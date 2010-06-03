import os, sys
from os.path import join, split, abspath
import numpy as np
from numpy.distutils.system_info import get_info
fftw_info = get_info('fftw3')

nthreads = 1
if sys.platform == 'darwin':
    try:
        nthreads = int(os.popen('sysctl -n hw.activecpu').read().strip())
    except:
        pass
elif sys.platform != 'win32':
    try:
        nthreads = os.sysconf('SC_NPROCESSORS_ONLN')
    except:
        pass
if fftw_info and sys.platform != 'win32':
    def export_extension(build=False):
        from scipy.weave import ext_tools
        from scipy.weave.converters import blitz as blitz_conv

        # call_code needs python args ['a','b','adims','fft_sign','shift','inplace']
        call_code = """
        if(inplace) {
          SCL_TYPE_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
        } else {
          SCL_TYPE_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
        }
        """

        fft_code = open(join(split(__file__)[0],'src/blitz_ffts.cc')).read()

        scl_types = dict(( (np.dtype('F'), 'cfloat'),
                           (np.dtype('D'), 'cdouble') ))

        blitz_ranks = range(1,12)

        fftw_libs = ['fftw3', 'fftw3f']
        defines = []

        if nthreads > 1:
            fftw_libs += ['pthread']
            defines += [('THREADED', None),
                        ('NTHREADS', nthreads),
                        ('BZ_THREADSAFE', None)]

        ext_funcs = []
        fft_sign = -1
        shift = 1
        inplace = 0
        for scl_type in scl_types:
            for rank in blitz_ranks:
                shape = (1,) * rank
                a = np.empty(shape, scl_type)
                b = np.empty(shape, scl_type)
                adims = np.array([0], 'i')
                c_scl_type = scl_types[scl_type]
                fcode = call_code.replace('SCL_TYPE', c_scl_type)
                fname = '_fft_%s_%d'%(scl_type.char, rank)

                ext_funcs.append(ext_tools.ext_function(fname, fcode,
                                                        ['a', 'b', 'adims',
                                                         'fft_sign', 'shift',
                                                         'inplace'],
                                                        type_converters=blitz_conv))
        mod = ext_tools.ext_module('fft_ext')
        for func in ext_funcs:
            mod.add_function(func)
        mod.customize.add_support_code(fft_code)
        mod.customize.set_compiler('gcc')
        mod.customize.add_header('<fftw3.h>')
        kw = {'libraries':fftw_libs,
              'include_dirs':fftw_info['include_dirs'] + [np.get_include()],
              'library_dirs':fftw_info['library_dirs'],
              'define_macros':defines}
        loc = split(__file__)[0]
        if build:
            mod.compile(location=loc, **kw)
        else:
            # this also generates the cpp file
            ext = mod.setup_extension(location=loc, **kw)
            ext.name = 'nutmeg.fftmod.fft_ext'
            return ext

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('fftmod', parent_package, top_path)

    config.add_data_dir('tests')

    if 'export_extension' in globals():
        fft_ext = export_extension(build=False)
        config.ext_modules.append(fft_ext)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
    
