import tempfile, os
import nose.tools as nt
import numpy as np
import numpy.testing as npt

from nutmeg.external import decotest

# import all from the module to test
from nutmeg.utils import *

@decotest.parametric
def test_cmap_to_array_roundtrip():
    aff = np.random.randn(4,4)
    aff[3] = np.array([0,0,0,1])
    cmap = ni_api.AffineTransform.from_params(
        'ijk', 'xyz', aff
        )
    cmap2 = cmap_from_array( parameterize_cmap( cmap ) )

    function_domain_same = cmap.function_domain.dtype == \
                           cmap2.function_domain.dtype
    function_domain_same&= cmap.function_domain.name == \
                           cmap2.function_domain.name
    function_domain_same&= cmap.function_domain.coord_names == \
                        cmap2.function_domain.coord_names
    yield function_domain_same, 'input coords differ'
    
    function_range_same = cmap.function_range.dtype == \
                          cmap2.function_range.dtype
    function_range_same&= cmap.function_range.name == \
                          cmap2.function_range.name
    function_range_same&= cmap.function_range.coord_names == \
                        cmap2.function_range.coord_names
    yield function_range_same, 'output coords differ'

    yield (cmap.affine==cmap2.affine).all(), 'affines differ'

class A(array_pickler_mixin):

    _argnames = ['str', 'arr', 'list', 'int', 'float']

class B(array_pickler_mixin):

    _argnames = ['arr', 'atype']
    # these need to also be identical to the name keyword args
    _kwnames = ['opt1', 'opt2']

def apickler_equality(p1, p2):

    if type(p1)!=type(p2):
        return False, str(type(p1))+' neq '+str(type(p2))

    attr_names = p1._argnames + p1._kwnames
    for name in attr_names:
        attr1 = getattr(p1, name)
        attr2 = getattr(p2, name)
        err_msg = str(attr1)+' neq '+str(attr2)
        if issubclass(type(attr1), array_pickler_mixin):
            if not apickler_equality(attr1, attr2):
                return False, err_msg
        elif issubclass(type(attr1), np.ndarray):
            if not (attr1==attr2).all():
                return False, err_msg
        elif attr1 != attr2:
            return False, err_msg
    return True, None

def test_complex_object_roundtrip():
    a = A('asdf', np.random.randn(10), [8,3,1,0,4], 100, 4.25)
    b = B(np.arange(20), a, opt1='blah blah blah')

    f = tempfile.mktemp(suffix='.npy')
    b.save(f)
    arr = np.load(f)
    os.unlink(f)
    b2 = B.from_array(arr)
    eq, msg = apickler_equality(b, b2)
    assert eq, msg
    
class HasSequences(array_pickler_mixin):
    _argnames = ['a_list', 'a_tuple']

def test_array_pickler_with_sequences():
    foo = HasSequences( [['a', None, 'c'], [1, 'three']],
                        (3,2,4, [4, 3] ) )

    f = tempfile.mktemp(suffix='.npy')
    foo.save(f)

    foo2 = HasSequences.load(f)
    yield nt.assert_true, foo.a_list==foo2.a_list, 'List roundtrip failed'
    yield nt.assert_true, foo.a_tuple==foo2.a_tuple, 'Tuple roundtrip failed'
