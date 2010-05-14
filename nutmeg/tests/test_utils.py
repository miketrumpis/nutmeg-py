import tempfile, os
import numpy as np
import numpy.testing as npt

from nutmeg.external import decotest

# import all from the module to test
from nutmeg.utils import *

@decotest.ipdoctest
def test_voxel_index_list():
    """
>>> voxel_index_list((2,3), order='ijk')
array([[0, 0],
       [1, 0],
       [0, 1],
       [1, 1],
       [0, 2],
       [1, 2]])
>>> voxel_index_list((2,3), order='kji')
array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2]])
    """

@decotest.ipdoctest
def test_coord_list_to_mgrid():
    """
>>> coords_list = voxel_index_list((2,3), order='ijk')
>>> mgrid = coord_list_to_mgrid(coords_list, (2,3), order='ijk')
>>> mgrid
array([[[0, 0, 0],
        [1, 1, 1]],
<BLANKLINE>
       [[0, 1, 2],
        [0, 1, 2]]])
>>> coords_list = voxel_index_list((2,3), order='kji')
>>> mgrid = coord_list_to_mgrid(coords_list, (2,3), order='kji')
>>> mgrid
array([[[0, 0, 0],
        [1, 1, 1]],
<BLANKLINE>
       [[0, 1, 2],
        [0, 1, 2]]])

    """

@decotest.parametric
def test_cmap_to_array_roundtrip():
    cmap = ni_api.Affine.from_params('ijk', 'xyz', np.random.randn(4,4))
    cmap2 = cmap_from_array( parameterize_cmap( cmap ) )

    input_coords_same = cmap.input_coords.dtype == cmap2.input_coords.dtype
    input_coords_same&= cmap.input_coords.name == cmap2.input_coords.name
    input_coords_same&= cmap.input_coords.coord_names == \
                        cmap2.input_coords.coord_names
    yield input_coords_same, 'input coords differ'
    
    output_coords_same = cmap.output_coords.dtype == cmap2.output_coords.dtype
    output_coords_same&= cmap.output_coords.name == cmap2.output_coords.name
    output_coords_same&= cmap.output_coords.coord_names == \
                        cmap2.output_coords.coord_names
    yield output_coords_same, 'output coords differ'

    yield (cmap.affine==cmap2.affine).all(), 'affines differ'

class A(array_pickler_mixin):

    _argnames = ['str', 'arr', 'list', 'int', 'float']

    def __init__(self, a_string, an_array, a_list, an_int, a_float):
        self.str = a_string
        self.arr = an_array
        self.list = a_list
        self.int = an_int
        self.float = a_float

class B(array_pickler_mixin):

    _argnames = ['arr', 'atype']
    # these need to also be identical to the name keyword args
    _kwnames = ['opt1', 'opt2']

    def __init__(self, an_array, typeA, opt1=None, opt2=None):
        self.arr = an_array
        self.atype = typeA
        self.opt1 = opt1
        self.opt2 = opt2

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
    
