from glob import glob
from os import path

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('vis', parent_package, top_path)

    # do NOT add tests dir.. the only test has a hard link to
    # my home directory
##     config.add_data_dir('tests')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
    
