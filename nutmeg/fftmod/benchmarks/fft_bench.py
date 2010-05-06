from numpy.testing import *
from recon.fftmod import fft1, ifft1, fft2, ifft2
from recon.util import checkerline, checkerboard
import numpy as np
from numpy import double, single, cdouble, csingle
from numpy.random import rand as random
import sys, time

class TestFFT1(TestCase):    
    def bench_fft1_time(self):
        from numpy.fft import fft as numpy_fft
        from scipy.fftpack import fft as scipy_fft
        print
        print '   1D Double Precision Fast Fourier Transform'
        print '================================================='
        print '      |   complex input    '
        print '-------------------------------------------------'
        print ' size |  recon  |  numpy  |  scipy  |'
        print '-------------------------------------------------'
        for size,repeat in [(100,7000),(1000,2000),
                            (256,10000),
                            (512,10000),
                            (1024,1000),
                            (2048,1000),
                            (2048*2,500),
                            (2048*4,500),
                            ]:
            print '%5s' % size,
            sys.stdout.flush()

            x = random(repeat, size).astype(cdouble) + \
                random(repeat, size).astype(cdouble)*1j
            tr0 = time.time()
            y = fft1(x, shift=False)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()
            tn0 = time.time()
            ny = numpy_fft(x)
            tnf = time.time()
            assert_array_almost_equal(ny,y)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()
            ts0 = time.time()
            sy = scipy_fft(x)
            tsf = time.time()
            assert_array_almost_equal(sy,y)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        sys.stdout.flush()
        
        print
        print '   1D Double Precision Fast Fourier Transform'
        print '================================================='
        print '      |   complex input shifted   '
        print '-------------------------------------------------'
        print ' size |  recon  |  numpy  |  scipy  |'
        print '-------------------------------------------------'
        for size,repeat in [(100,7000),(1000,2000),
                            (256,10000),
                            (512,10000),
                            (1024,1000),
                            (2048,1000),
                            (2048*2,500),
                            (2048*4,500),
                            ]:
            chk = checkerline(size).astype(double)
            print '%5s' % size,
            sys.stdout.flush()

            x = random(repeat, size).astype(cdouble) + \
                random(repeat, size).astype(cdouble)*1j
            tr0 = time.time()
            y = fft1(x, shift=True)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()
            tn0 = time.time()
            ny = chk*numpy_fft(chk*x)
            tnf = time.time()
            assert_array_almost_equal(ny,y)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()
            ts0 = time.time()
            sy = chk*scipy_fft(chk*x)
            tsf = time.time()
            assert_array_almost_equal(sy,y)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        sys.stdout.flush()

        print
        print '   1D Single Precision Fast Fourier Transform'
        print '================================================='
        print '      |   complex input    '
        print '-------------------------------------------------'
        print ' size |  recon  |  numpy  |  scipy*  |'
        print '-------------------------------------------------'
        for size,repeat in [(100,7000),(1000,2000),
                            (256,10000),
                            (512,10000),
                            (1024,1000),
                            (2048,1000),
                            (2048*2,500),
                            (2048*4,500),
                            ]:
            print '%5s' % size,
            sys.stdout.flush()

            x = random(repeat, size).astype(csingle) + \
                random(repeat, size).astype(csingle)*1j
            tr0 = time.time()
            y = fft1(x, shift=False)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()
            tn0 = time.time()
            ny = numpy_fft(x)
            tnf = time.time()
            assert_array_almost_equal(ny,y, decimal=2)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()
            ts0 = time.time()
            sy = scipy_fft(x.astype(cdouble)).astype(csingle)
            tsf = time.time()
            assert_array_almost_equal(sy,y, decimal=2)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        print "(* casted float->FT{double}->float)"

        sys.stdout.flush()
        print
        print '   1D Single Precision Fast Fourier Transform'
        print '================================================='
        print '      |   complex input shifted   '
        print '-------------------------------------------------'
        print ' size |  recon  |  numpy  |  scipy*  |'
        print '-------------------------------------------------'
        for size,repeat in [(100,7000),(1000,2000),
                            (256,10000),
                            (512,10000),
                            (1024,1000),
                            (2048,1000),
                            (2048*2,500),
                            (2048*4,500),
                            ]:
            chk = checkerline(size).astype(single)
            print '%5s' % size,
            sys.stdout.flush()

            x = random(repeat, size).astype(csingle) + \
                random(repeat, size).astype(csingle)*1j
            tr0 = time.time()
            y = fft1(x, shift=True)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()
            tn0 = time.time()
            ny = chk*numpy_fft(chk*x)
            tnf = time.time()
            assert_array_almost_equal(ny,y, decimal=2)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()
            ts0 = time.time()
            sy = chk*(scipy_fft((chk*x).astype(cdouble))).astype(csingle)
            tsf = time.time()
            assert_array_almost_equal(sy,y, decimal=2)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        print "(* casted float->FT{double}->float)"
        sys.stdout.flush()

class TestIFFT1(TestCase):
    def bench_ifft1_time(self):
        from numpy.fft import ifft as numpy_ifft
        from scipy.fftpack import ifft as scipy_ifft
        print
        print ' 1D Double Precision (I)Fast Fourier Transform'
        print '================================================='
        print '      |   complex input    '
        print '-------------------------------------------------'
        print ' size |  recon  |  numpy  |  scipy  |'
        print '-------------------------------------------------'
        for size,repeat in [(100,7000),(1000,2000),
                            (256,10000),
                            (512,10000),
                            (1024,1000),
                            (2048,1000),
                            (2048*2,500),
                            (2048*4,500),
                            ]:
            print '%5s' % size,
            sys.stdout.flush()

            x = random(repeat, size).astype(cdouble) + \
                random(repeat, size).astype(cdouble)*1j
            tr0 = time.time()
            y = ifft1(x, shift=False)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()
            tn0 = time.time()
            ny = numpy_ifft(x)
            tnf = time.time()
            assert_array_almost_equal(ny,y)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()
            ts0 = time.time()
            sy = scipy_ifft(x)
            tsf = time.time()
            assert_array_almost_equal(sy,y)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        sys.stdout.flush()

        print
        print ' 1D Double Precision (I)Fast Fourier Transform'
        print '================================================='
        print '      |   complex input shifted    '
        print '-------------------------------------------------'
        print ' size |  recon  |  numpy  |  scipy  |'
        print '-------------------------------------------------'
        for size,repeat in [(100,7000),(1000,2000),
                            (256,10000),
                            (512,10000),
                            (1024,1000),
                            (2048,1000),
                            (2048*2,500),
                            (2048*4,500),
                            ]:
            chk = checkerline(size).astype(double)
            print '%5s' % size,
            sys.stdout.flush()

            x = random(repeat, size).astype(cdouble) + \
                random(repeat, size).astype(cdouble)*1j
            tr0 = time.time()
            y = ifft1(x, shift=True)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()
            tn0 = time.time()
            ny = chk*numpy_ifft(chk*x)
            tnf = time.time()
            assert_array_almost_equal(ny,y)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()
            ts0 = time.time()
            sy = chk*scipy_ifft(chk*x)
            tsf = time.time()
            assert_array_almost_equal(sy,y)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        sys.stdout.flush()
        
        print
        print ' 1D Single Precision (I)Fast Fourier Transform'
        print '================================================='
        print '      |   complex input    '
        print '-------------------------------------------------'
        print ' size |  recon  |  numpy  |  scipy*  |'
        print '-------------------------------------------------'
        for size,repeat in [(100,7000),(1000,2000),
                            (256,10000),
                            (512,10000),
                            (1024,1000),
                            (2048,1000),
                            (2048*2,500),
                            (2048*4,500),
                            ]:
            print '%5s' % size,
            sys.stdout.flush()

            x = random(repeat, size).astype(csingle) + \
                random(repeat, size).astype(csingle)*1j
            tr0 = time.time()
            y = ifft1(x, shift=False)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()
            tn0 = time.time()
            ny = numpy_ifft(x)
            tnf = time.time()
            assert_array_almost_equal(ny,y, decimal=2)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()
            ts0 = time.time()
            sy = scipy_ifft(x.astype(cdouble)).astype(csingle)
            tsf = time.time()
            assert_array_almost_equal(sy,y, decimal=2)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        print "(* casted float->FT{double}->float)"
        sys.stdout.flush()

        print
        print ' 1D Single Precision (I)Fast Fourier Transform'
        print '================================================='
        print '      |   complex input shifted   '
        print '-------------------------------------------------'
        print ' size |  recon  |  numpy  |  scipy*  |'
        print '-------------------------------------------------'
        for size,repeat in [(100,7000),(1000,2000),
                            (256,10000),
                            (512,10000),
                            (1024,1000),
                            (2048,1000),
                            (2048*2,500),
                            (2048*4,500),
                            ]:
            chk = checkerline(size).astype(single)
            print '%5s' % size,
            sys.stdout.flush()

            x = random(repeat, size).astype(csingle) + \
                random(repeat, size).astype(csingle)*1j
            tr0 = time.time()
            y = ifft1(x, shift=True)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()
            tn0 = time.time()
            ny = chk*numpy_ifft(chk*x)
            tnf = time.time()
            assert_array_almost_equal(ny,y, decimal=2)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()
            ts0 = time.time()
            sy = chk*(scipy_ifft((chk*x).astype(cdouble))).astype(csingle)
            tsf = time.time()
            assert_array_almost_equal(sy,y, decimal=2)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        print "(* casted float->FT{double}->float)"
        sys.stdout.flush()

class TestFFT2(TestCase):
    def bench_fft2_time(self):
        from numpy.fft import fftn as numpy_fftn
        from scipy.fftpack import fftn as scipy_fftn
        print
        print '    2D Double Precision Fast Fourier Transform'
        print '==================================================='
        print '          |   complex input    '
        print '---------------------------------------------------'
        print '   size   |  recon  |  numpy  |  scipy  |'
        print '---------------------------------------------------'
        for size,repeat in [((100,100),100),((1000,100),7),
                            ((256,256),10),
                            ((512,512),3),
                            ]:
            print '%9s' % ('%sx%s'%size),
            sys.stdout.flush()
            x = random(repeat, *size).astype(cdouble) + \
                random(repeat, *size).astype(cdouble)*1j
            tr0 = time.time()
            y = fft2(x, shift=False)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()

            tn0 = time.time()
            ny = numpy_fftn(x, axes=(-2,-1))
            tnf = time.time()
            assert_array_almost_equal(ny,y)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()

            ts0 = time.time()
            sy = scipy_fftn(x, axes=(-2,-1))
            tsf = time.time()
            assert_array_almost_equal(sy,y)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        sys.stdout.flush()

        print
        print '    2D Double Precision Fast Fourier Transform'
        print '==================================================='
        print '          |   complex input shifted   '
        print '---------------------------------------------------'
        print '   size   |  recon  |  numpy  |  scipy  |'
        print '---------------------------------------------------'
        for size,repeat in [((100,100),100),((1000,100),7),
                            ((256,256),10),
                            ((512,512),3),
                            ]:
            chk = checkerboard(*size).astype(double)
            print '%9s' % ('%sx%s'%size),
            sys.stdout.flush()
            x = random(repeat, *size).astype(cdouble) + \
                random(repeat, *size).astype(cdouble)*1j
            tr0 = time.time()
            y = fft2(x, shift=True)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()


        print
        print '    2D Single Precision Fast Fourier Transform'
        print '==================================================='
        print '          |   complex input    '
        print '---------------------------------------------------'
        print '   size   |  recon  |  numpy  |  scipy*  |'
        print '---------------------------------------------------'
        for size,repeat in [((100,100),100),((1000,100),7),
                            ((256,256),10),
                            ((512,512),3),
                            ]:
            print '%9s' % ('%sx%s'%size),
            sys.stdout.flush()
            x = random(repeat, *size).astype(csingle) + \
                random(repeat, *size).astype(csingle)*1j
            tr0 = time.time()
            y = fft2(x, shift=False)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()

            tn0 = time.time()
            ny = numpy_fftn(x, axes=(-2,-1))
            tnf = time.time()
            assert_array_almost_equal(ny,y, decimal=2)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()

            ts0 = time.time()
            sy = scipy_fftn(x.astype(cdouble), axes=(-2,-1)).astype(csingle)
            tsf = time.time()
            assert_array_almost_equal(sy,y, decimal=2)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        print "(* casted float->FT{double}->float)"
        sys.stdout.flush()

        print
        print '    2D Single Precision Fast Fourier Transform'
        print '==================================================='
        print '          |   complex input shifted   '
        print '---------------------------------------------------'
        print '   size   |  recon  |  numpy  |  scipy*  |'
        print '---------------------------------------------------'
        for size,repeat in [((100,100),100),((1000,100),7),
                            ((256,256),10),
                            ((512,512),3),
                            ]:
            chk = checkerboard(*size).astype(single)
            print '%9s' % ('%sx%s'%size),
            sys.stdout.flush()
            x = random(repeat, *size).astype(csingle) + \
                random(repeat, *size).astype(csingle)*1j
            tr0 = time.time()
            y = fft2(x, shift=True)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()

            tn0 = time.time()
            ny = chk*numpy_fftn(chk*x, axes=(-2,-1))
            tnf = time.time()
            assert_array_almost_equal(ny,y, decimal=2)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()

            ts0 = time.time()
            sy = chk*(scipy_fftn((chk*x).astype(cdouble), axes=(-2,-1)).astype(csingle))
            tsf = time.time()
            assert_array_almost_equal(sy,y, decimal=2)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        print "(* casted float->FT{double}->float)"
        sys.stdout.flush()

class TestIFFT2(TestCase):
    def bench_ifft2_time(self):
        from numpy.fft import ifftn as numpy_ifftn
        from scipy.fftpack import ifftn as scipy_ifftn
        print
        print '    2D Double Precision (I)Fast Fourier Transform'
        print '==================================================='
        print '          |   complex input    '
        print '---------------------------------------------------'
        print '   size   |  recon  |  numpy  |  scipy  |'
        print '---------------------------------------------------'
        for size,repeat in [((100,100),100),((1000,100),7),
                            ((256,256),10),
                            ((512,512),3),
                            ]:
            print '%9s' % ('%sx%s'%size),
            sys.stdout.flush()
            x = random(repeat, *size).astype(cdouble) + \
                random(repeat, *size).astype(cdouble)*1j
            tr0 = time.time()
            y = ifft2(x, shift=False)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()

            tn0 = time.time()
            ny = numpy_ifftn(x, axes=(-2,-1))
            tnf = time.time()
            assert_array_almost_equal(ny,y)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()

            ts0 = time.time()
            sy = scipy_ifftn(x, axes=(-2,-1))
            tsf = time.time()
            assert_array_almost_equal(sy,y)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        sys.stdout.flush()

        print
        print '    2D Double Precision (I)Fast Fourier Transform'
        print '==================================================='
        print '          |   complex input shifted   '
        print '---------------------------------------------------'
        print '   size   |  recon  |  numpy  |  scipy  |'
        print '---------------------------------------------------'
        for size,repeat in [((100,100),100),((1000,100),7),
                            ((256,256),10),
                            ((512,512),3),
                            ]:
            chk = checkerboard(*size).astype(double)
            print '%9s' % ('%sx%s'%size),
            sys.stdout.flush()
            x = random(repeat, *size).astype(cdouble) + \
                random(repeat, *size).astype(cdouble)*1j
            tr0 = time.time()
            y = ifft2(x, shift=True)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()

            tn0 = time.time()
            ny = chk*numpy_ifftn(chk*x, axes=(-2,-1))
            tnf = time.time()
            assert_array_almost_equal(ny,y)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()

            ts0 = time.time()
            sy = chk*scipy_ifftn(chk*x, axes=(-2,-1))
            tsf = time.time()
            assert_array_almost_equal(sy,y)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        sys.stdout.flush()

        print
        print '    2D Single Precision (I)Fast Fourier Transform'
        print '==================================================='
        print '          |   complex input    '
        print '---------------------------------------------------'
        print '   size   |  recon  |  numpy  |  scipy*  |'
        print '---------------------------------------------------'
        for size,repeat in [((100,100),100),((1000,100),7),
                            ((256,256),10),
                            ((512,512),3),
                            ]:
            print '%9s' % ('%sx%s'%size),
            sys.stdout.flush()
            x = random(repeat, *size).astype(csingle) + \
                random(repeat, *size).astype(csingle)*1j
            tr0 = time.time()
            y = ifft2(x, shift=False)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()

            tn0 = time.time()
            ny = numpy_ifftn(x, axes=(-2,-1))
            tnf = time.time()
            assert_array_almost_equal(ny,y, decimal=2)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()

            ts0 = time.time()
            sy = scipy_ifftn(x.astype(cdouble), axes=(-2,-1)).astype(csingle)
            tsf = time.time()
            assert_array_almost_equal(sy,y, decimal=2)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        print "(* casted float->FT{double}->float)"
        sys.stdout.flush()

        print
        print '    2D Single Precision (I)Fast Fourier Transform'
        print '==================================================='
        print '          |   complex input shifted   '
        print '---------------------------------------------------'
        print '   size   |  recon  |  numpy  |  scipy*  |'
        print '---------------------------------------------------'
        for size,repeat in [((100,100),100),((1000,100),7),
                            ((256,256),10),
                            ((512,512),3),
                            ]:
            chk = checkerboard(*size).astype(single)
            print '%9s' % ('%sx%s'%size),
            sys.stdout.flush()
            x = random(repeat, *size).astype(csingle) + \
                random(repeat, *size).astype(csingle)*1j
            tr0 = time.time()
            y = ifft2(x, shift=True)
            trf = time.time()
            print '|%8.2f' % (trf-tr0),
            sys.stdout.flush()

            tn0 = time.time()
            ny = chk*numpy_ifftn(chk*x, axes=(-2,-1))
            tnf = time.time()
            assert_array_almost_equal(ny,y, decimal=2)
            print '|%8.2f' % (tnf-tn0),
            sys.stdout.flush()

            ts0 = time.time()
            sy = chk*(scipy_ifftn((chk*x).astype(cdouble), axes=(-2,-1)).astype(csingle))
            tsf = time.time()
            assert_array_almost_equal(sy,y, decimal=2)
            print '|%8.2f' % (tsf-ts0),
            sys.stdout.flush()

            print ' (secs for %s calls)' % (repeat)
        print "(* casted float->FT{double}->float)"
        sys.stdout.flush()
