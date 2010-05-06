from nutmeg.fftmod import fft1, ifft1, fft2, ifft2, _fftn, _ifftn
import numpy as np

def direct_dft(v):
    assert v.dtype.char.isupper()
    L = v.shape[0]
    Lax = np.linspace(0,L,L,endpoint=False)
    basis = np.exp((Lax[:,None] * -2j*np.pi*Lax/L)).astype(v.dtype)
    w = np.dot(basis, v.reshape(L, np.prod(v.shape[1:])))
    w.shape = v.shape
    return w

def direct_idft(v):
    assert v.dtype.char.isupper()
    L = v.shape[0]
    Lax = np.linspace(0,L,L,endpoint=False)
    basis = np.exp((Lax[:,None] * 2j*np.pi*Lax/L)).astype(v.dtype)
    w = np.dot(basis, v.reshape(L, np.prod(v.shape[1:])))
    w.shape = v.shape
    return w
    
def direct_dft_centered(v):
    assert v.dtype.char.isupper()
    L = v.shape[0]
    grid = np.linspace(-L/2,L/2,num=L,endpoint=False)
    basis = np.exp((grid[:,None] * -2j*np.pi*grid/L)).astype(v.dtype)
    w = np.dot(basis, v.reshape(L, np.prod(v.shape[1:])))
    w.shape = v.shape
    return w

def direct_idft_centered(v):
    assert v.dtype.char.isupper()
    L = v.shape[0]
    grid = np.linspace(-L/2,L/2,num=L,endpoint=False)
    basis = np.exp((grid[:,None] * 2j*np.pi*grid/L)).astype(v.dtype)
    w = np.dot(basis, v.reshape(L, np.prod(v.shape[1:])))
    w.shape = v.shape
    return w

def checkerline(cols):
    return np.ones(cols) - 2*(np.arange(cols)%2)

def checkerboard(rows, cols):
    return np.outer(checkerline(rows), checkerline(cols))

## def reference_fftn(a, axes=(0,), shift=True):
##     a_s = np.fft.fftshift(a,axes=axes) if shift else a
##     b = np.fft.fftn(a_s, axes=axes)
##     return np.fft.fftshift(b, axes=axes) if shift else b

def reference_fftn(a, axes=(0,), shift=True):
    dft_func = direct_dft_centered if shift else direct_dft
    a_dft = a.copy()
    for ax in axes:
        b = np.rollaxis(a_dft, ax)
        b[:] = dft_func(b)
    return a_dft

## def reference_ifftn(a, axes=(0,), shift=True):
##     a_s = np.fft.fftshift(a,axes=axes) if shift else a
##     b = np.fft.ifftn(a_s, axes=axes)
##     return np.fft.fftshift(b, axes=axes) if shift else b

def reference_ifftn(a, axes=(0,), shift=True):
    dft_func = direct_idft_centered if shift else direct_idft
    a_dft = a.copy()
    for ax in axes:
        b = np.rollaxis(a_dft, ax)
        b[:] = dft_func(b)
    return a_dft


# decorator goes with the TestFFT class
def run_toggled_kwargs(meth):
    def run_parameterized(obj):
        for dt_char in ['F', 'D']:
            for shift_val in [0, 1]:
                obj.dt = np.dtype(dt_char)
                meth(obj, shift=shift_val)
    run_parameterized.func_name = meth.func_name
    return run_parameterized

class TestFFT(np.testing.TestCase):
    def setUp(self):
        # a 32Hz complex exponential (sampling rate = 128Hz)
        self.dt = np.dtype('F')
        self.sgrid = np.linspace(0, 1, num=128, endpoint=False)
        self.ref_1D_32hz = np.exp(2j*np.pi*32*self.sgrid)
        self.ref_2D_grating = np.exp( (2j*np.pi*13*self.sgrid)[:,None] + \
                                      (2j*np.pi*4*self.sgrid)[None,:] )
        
        self.rand_3d = (np.random.randn(40, 50, 60) + \
                        1j*np.random.randn(40, 50, 60))
        self.comp = np.testing.assert_almost_equal

    def nrg_comp(self, a, b, err_msg=''):
        e = a.flat[:]-b.flat[:]
        self.comp(np.dot(e,e.conj()).real, 0, err_msg=err_msg)
        
    @run_toggled_kwargs
    def test_simple_1D_fft(self, **kwargs):
        shift = kwargs.get('shift', 0)
        c = self.ref_1D_32hz.astype(self.dt)
        c2 = self.ref_1D_32hz.astype(self.dt)
        _fftn(c, axes=(0,), shift=shift, inplace=True)
        c_np = reference_fftn(c2, axes=(0,), shift=shift)
        self.comp(np.dot(c,c.conj()).real,
                  np.dot(c_np, c_np.conj()).real)
        self.nrg_comp(c, c_np)
        # analytically, the DFT indexed from 0,127 of s
        # is a weighted delta at k=32
        C_a = np.zeros_like(c)
        C_a[32 + shift*64] = 128.
        self.comp(c, C_a, err_msg='dtype = %s'%self.dt.char)
        self.nrg_comp(c, C_a)        

    @run_toggled_kwargs
    def test_simple_2D_fft(self, **kwargs):
        shift = kwargs.get('shift', 0)
        c = self.ref_2D_grating.astype(self.dt)
        c2 = self.ref_2D_grating.astype(self.dt)
        _fftn(c, axes=(0,1), shift=shift, inplace=True)
        c_np = reference_fftn(c2, axes=(0,1), shift=shift)
        self.comp(np.dot(c.flat[:],c.flat[:].conj()).real,
                  np.dot(c_np.flat[:], c_np.flat[:].conj()).real)
        self.nrg_comp(c, c_np)
        # analytically, the DFT indexed from 0,127 of s
        # is a weighted delta at i=13, j=4
        C_a = np.zeros_like(c)
        C_a[13 + shift*64, 4 + shift*64] = 128**2
        self.comp(c, C_a, err_msg='dtype = %s'%self.dt.char)
        self.nrg_comp(c, C_a)
    
    @run_toggled_kwargs
    def test_simple_multi_fft(self, **kwargs):
        shift = kwargs.get('shift', 0)        
        c = np.outer(np.ones(64), self.ref_1D_32hz).astype(self.dt)
        c2 = np.outer(np.ones(64), self.ref_1D_32hz).astype(self.dt)
        ct = c.copy().T
        ct2 = c.copy().T
        _fftn(c, axes=(-1,), inplace=True, shift=shift)
        _fftn(ct, axes=(0,), inplace=True, shift=shift)
        c_np = reference_fftn(c2, axes=(-1,), shift=shift)
        ct_np = reference_fftn(ct2, axes=(0,), shift=shift)
        self.nrg_comp(c, c_np, err_msg='dtype = %s'%self.dt.char)
        self.nrg_comp(ct, ct_np, err_msg='transpose dtype=%s'%self.dt.char)
        
    @run_toggled_kwargs        
    def test_strided_1d_fft(self, **kwargs):
        shift = kwargs.get('shift', 0)        
        r1 = self.rand_3d.copy().transpose(0,2,1).astype(self.dt)
        r1_2 = self.rand_3d.copy().transpose(0,2,1).astype(self.dt)
        r2 = self.rand_3d.copy().transpose(1,0,2).astype(self.dt)
        r2_2 = self.rand_3d.copy().transpose(1,0,2).astype(self.dt)        

        _fftn(r1, axes=(0,), inplace=True, shift=shift)
        _fftn(r2, axes=(1,), inplace=True, shift=shift)
        r1_np = reference_fftn(r1_2, axes=(0,), shift=shift)
        r2_np = reference_fftn(r2_2, axes=(1,), shift=shift)

        self.nrg_comp(r1, r1_np,
                      err_msg='axis0 dtype = %s'%self.dt.char)
##         self.comp( (r1*r1.conj()).real.sum(),
##                    (r1_np*r1_np.conj()).real.sum(),
##                    err_msg='axis0 dtype = %s'%self.dt.char)
        self.nrg_comp(r2, r2_np,
                      err_msg='axis1 dtype = %s'%self.dt.char)
##         self.comp( (r2*r2.conj()).real.sum(),
##                    (r2_np*r2_np.conj()).real.sum(),
##                    err_msg='axis1 dtype = %s'%self.dt.char)

    @run_toggled_kwargs
    def test_strided_2d_fft(self, **kwargs):
        shift = kwargs.get('shift', 0)
        r1 = self.rand_3d.copy().transpose(0,2,1).astype(self.dt)
        r2 = self.rand_3d.copy().transpose(1,0,2).astype(self.dt)
        r1_2 = self.rand_3d.copy().transpose(0,2,1).astype(self.dt)
        r2_2 = self.rand_3d.copy().transpose(1,0,2).astype(self.dt)

        _fftn(r1, axes=(0,2), inplace=True, shift=shift)
        _fftn(r2, axes=(1,2), inplace=True, shift=shift)
        r1_np = reference_fftn(r1_2, axes=(0,2), shift=shift)
        r2_np = reference_fftn(r2_2, axes=(1,2), shift=shift)

##         self.comp( (r1*r1.conj()).real.sum(),
##                    (r1_np*r1_np.conj()).real.sum(),
##                    err_msg='dtype = %s'%self.dt.char )
        self.nrg_comp(r1, r1_np,
                      err_msg='dtype = %s'%self.dt.char )
##         self.comp( (r2*r2.conj()).real.sum(),
##                    (r2_np*r2_np.conj()).real.sum(),
##                    err_msg='dtype = %s'%self.dt.char )
        err = r2.flat[:] - r2_np.flat[:]
        self.nrg_comp(r2, r2_np,
                      err_msg='dtype = %s'%self.dt.char )

    @run_toggled_kwargs
    def test_roundtrip_outofplace(self, **kwargs):
        shift = kwargs.get('shift', 0)

        grid = np.arange(128)
        mu = 43.
        stdv = 3.
        g = (np.exp(-(grid-mu)**2 / (2*stdv**2)) / (2*np.pi*stdv**2)).astype(self.dt)
        g2 = g.copy()
        g_bkp = g.copy()
##         gw = np.empty_like(g)
##         grt = np.empty_like(g)

        gw = _fftn(g, inplace=False, shift=shift)
        gw_np = reference_fftn(g2, axes=(0,), shift=shift)
        grt = _ifftn(gw, inplace=False, shift=shift)
        grt_np = reference_ifftn(gw, axes=(0,), shift=shift)


        # strong assertion that g is not altered at all
        assert (g==g_bkp).all(), 'out of place xform error, shift=%d'%shift
        self.comp(gw, gw_np,
                  err_msg='differs from numpy fft ref, shift=%d'%shift)
        self.nrg_comp(gw, gw_np)
        self.comp(g, grt,
                  err_msg='roundtrip transforms diverge, shift=%d'%shift)
        self.nrg_comp(g, grt)        

    @run_toggled_kwargs
    def test_roundtrip_inplace(self, **kwargs):
        shift = kwargs.get('shift', 0)
        grid = np.arange(128)
        mu = 43.
        stdv = 3.
        g = (np.exp(-(grid-mu)**2 / (2*stdv**2)) / (2*np.pi*stdv**2)).astype(self.dt)
        g2 = g.copy()
        g_bkp = g.copy()
        _fftn(g, inplace=True, shift=shift)
        gw_np = reference_fftn(g2, shift=shift, axes=(0,))
        
        self.comp(g, gw_np,
                  err_msg='differs from numpy fft ref, shift=%d'%shift)
        self.nrg_comp(g, gw_np)
        _ifftn(g, inplace=True, shift=shift)
        self.comp(g_bkp, g,
                  err_msg='roundtrip transforms diverge, shift=%d'%shift)
        self.nrg_comp(g_bkp, g)        

        
        
    
