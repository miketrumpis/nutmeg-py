import numpy as np
import numpy.testing as npt
import nose.tools as nt

from nutmeg.core import tfbeam

def gen_tfbeam(fixed_comparison=None):
    # generate a TFBeam whose active-to-control ratio is equal
    # to its active component everywhere (control(x) := 1)
    from nutmeg.core.tests.test_beam import gen_beam
    b = gen_beam()
    tf_sig_active = b.sig[:,:,None] * np.ones(5)[None,None,:]
    if fixed_comparison is None:
        tf_sig_control = np.ones_like(tf_sig_active)
        tf_sig = (tf_sig_active, tf_sig_control)
    else:
        tf_sig = tf_sig_active

    bands = np.array( ( np.arange(5), np.arange(1,6) ) ).T
    n_time_pts = len(b.timepts)
    timewindow = np.array( ( np.arange(n_time_pts),
                             np.arange(1,n_time_pts+1) ) ).T
    tfb = tfbeam.TFBeam(
        b.voxelsize, b.voxels, b.srate, b.timepts,
        tf_sig, b.coreg, bands,
        timewindow, coordmap=b.coordmap,
        fixed_comparison=fixed_comparison)
    return tfb

def test_fixed_comp():
    b = gen_tfbeam(fixed_comparison='f db')
    yield nt.assert_true, b.uses == 'F dB', 'fixed comparison not translated'
    yield nt.assert_true, b.sig is b.s, 'sig array and s property not identical'
    st = b.signal_transforms
    yield nt.assert_true, len(st)==1 and 'F dB' in st, \
          'fixed comparison not in transform list'

def test_refix_comp():
    b = gen_tfbeam()
    b.fix_comparison('f raw')
    yield nt.assert_true, b.uses == 'F raw'
    yield nt.assert_true, b.sig is b.s
    st = b.signal_transforms
    yield nt.assert_true, len(st)==1 and 'F raw' in st

def test_failures():
    b = gen_tfbeam()
    b.fix_comparison('f raw')
    yield nt.assert_raises, RuntimeError, getattr, b, 'f_db', \
          'did not raise error on disabled attribute'
    yield nt.assert_true, hasattr(b, 'sig'), 'deleted the signal!!'
    b.fix_comparison('crazy')
    yield nt.assert_raises, RuntimeError, getattr, b, 'f_raw', \
          'did not raise error on disabled attribute'
    yield nt.assert_true, hasattr(b, 'sig'), 'deleted the signal!!'
    # now initialize from a fixed comparison
    b = gen_tfbeam(fixed_comparison='f db')
    # for the record, all attributes should be disabled (even f_db)
    yield nt.assert_raises, RuntimeError, getattr, b, 'f_db', \
          'did not raise error on disabled attribute'
    yield nt.assert_true, hasattr(b, 'sig'), 'deleted the signal!!'
    b.fix_comparison('fooooo') 
    yield nt.assert_raises, RuntimeError, getattr, b, 'f_db', \
          'did not raise error on disabled attribute'
    yield nt.assert_true, hasattr(b, 'sig'), 'deleted the signal!!'

    new_signal = np.zeros_like(b.s)
    # try to create a new TFBeam with no components, and no
    # fixed comparison.. should fail with ValueError
    yield nt.assert_raises, ValueError, b.from_new_data, new_signal

def test_translations():
    b = gen_tfbeam()
    b.uses = 'active power'
    yield nt.assert_true, b.uses=='Active Power', 'did not translate case'
    b.fix_comparison('f db')
    yield nt.assert_true, b.uses=='F dB', 'did not translate case'
    b = gen_tfbeam(fixed_comparison='f raw')
    yield nt.assert_true, b.uses == 'F raw', 'did not translate case'

def test_transforms():
    b = gen_tfbeam()
    active = b.active_power
    yield nt.assert_true, (b.control_power==1).all(), 'control power incorrect'
    yield nt.assert_true, (b.noise_power==0).all(), 'noise power incorrect'
    yield nt.assert_true, (b.f_raw == active).all(), 'f ratio incorrect'
    active = 10*np.log10(active)
    yield npt.assert_array_almost_equal, b.f_db, active
