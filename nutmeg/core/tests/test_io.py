import tempfile, os
import numpy as np

from nutmeg.external import decotest
from nutmeg.core.beam import Beam, MEG_coreg
from nutmeg.core.tfbeam import TFBeam

@decotest.parametric
def test_beam_io_runs():
    cr = MEG_coreg('asdf', 'fdsa', np.eye(4), np.eye(3))
    b = Beam(np.array([5.]*3),
                  np.random.randn(1000,3),
                  200.,
                  np.arange(100),
                  np.random.randn(1000),
                  cr)
    f = tempfile.mktemp(suffix='.npy')
    try:
        b.save(f)
        b2 = Beam.load(f)
        os.unlink(f)
        assert True
    except:
        os.unlink(f)
        assert False, 'simple I/O failed'
    
def test_tfbeam_io_runs():
    cr = MEG_coreg('asdf', 'fdsa', np.eye(4), np.eye(3))
    sig = np.array( (np.random.randn(50,10,3), np.random.randn(50,10,3)) )
    bands = np.arange(4)
    timepts = np.arange(10)
    timewin = np.array( [timepts - .5, timepts + .5] ).T
    b = TFBeam(np.array([5.]*3),
               np.random.randn(1000,3),
               200.,
               np.arange(100),
               np.random.randn(1000),
               cr,
               bands,
               timewin,
               fixed_comparison='F dB')
    f = tempfile.mktemp(suffix='.npy')
    try:
        b.save(f)
        b2 = TFBeam.load(f)
        os.unlink(f)
        assert True
    except:
        os.unlink(f)
        assert False, 'simple I/O failed'
