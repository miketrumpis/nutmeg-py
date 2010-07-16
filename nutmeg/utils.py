import numpy as np
import nipy.core.api as ni_api

from xipy.utils import voxel_index_list, coord_list_to_mgrid
from xipy.volume_utils import calc_grid_and_map

def parameterize_cmap(coordmap):
    dt = [('incoord', object), ('outcoord', object), ('affine', object)]
    a = np.zeros(1, dtype=dt)
    a['incoord'][0] = coordmap.function_domain.coord_names
    a['outcoord'][0] = coordmap.function_range.coord_names
    a['affine'][0] = coordmap.affine
    return a

def cmap_from_array(arr):
    aff = np.empty(arr['affine'][0].shape, 'd')
    aff[:] = arr['affine'][0]
    return ni_api.AffineTransform.from_params(arr['incoord'][0],
                                              arr['outcoord'][0],
                                              aff)

class array_pickler_mixin(object):
    """
    Can save-itself-as/load-itself-from a funky structured array.

    The structure is VERY brittle.

    This is accomplished by requiring each class that inherits this
    functionality to conform to the following rules:

    * define two class-level fields:
        _argnames -- the list of names of positional arguments
                     in the constructor
        _kwnames  -- the list of names of keyword arguments in
                     the constructor
    * The positional arguments must be listed in the correct order.
    * The names in both lists must be attributes of the class objects.
    * Each name in the kwnames list must be identical to both the
      corresponding class object attribute AND the keyword argument
      name in the class constructor.
    * All attributes named in the lists must be one of:
      * ndarray
      * array_pickler_mixin subclass
      * any type that can be reconstructed such that
        a == type(a)(asarray(a) )
        (satisfied by many built-in types)
      * any type that has special staticmethods '_array_from_{attr}'
        and '_reconstruct_{attr}' defined on the class. The first
        method creates an ndarray from an object of this type, and the
        second one reconstructs an object in a round-trip sense.

    Examples
    --------
    >>> class A(array_pickler_mixin):
    ...     _argnames = ['str', 'arr']
    ...     def __init__(self, a_string, an_array):
    ...         self.str = a_string
    ...         self.arr = an_array
    >>> class B(array_pickler_mixin):
    ...     _argnames = ['arr', 'atype']
    ...     # these need to also be identical to the name keyword args
    ...     _kwnames = ['opt1', 'opt2']
    ...     def __init__(self, an_array, typeA, opt1=None, opt2=None):
    ...         self.arr = an_array
    ...         self.atype = typeA
    ...         self.opt1 = opt1
    ...         self.opt2 = opt2
    >>> a = A('asdf', np.random.randn(10))
    >>> b = B(np.arange(20), a)
    >>> b.save('bsave') # actually, this don't work on the command-line
    >>> b2 = B.load('bsave.npy')
    """
    
    _argnames = []
    _kwnames = []

    @classmethod
    def _get_dtype(klass):
        type_list = [(n, object) for n in klass._argnames+klass._kwnames]
        type_list.append( ('conversion_lookup', object) )
        type_list.append( ('klass_string', object) )
        return np.dtype(type_list)


    def __init__(self, *args, **kwargs):
        if len(args) != len(self._argnames):
            raise ValueError('Not enough construtor arguments')
        self.__dict__.update( dict(zip(self._argnames, args)) )
        for nm in kwargs:
            if nm not in self._kwnames:
                raise ValueError('Unexpected keyword argument %s'%nm)
        for nm in self._kwnames:
            if nm not in kwargs:
                kwargs[nm] = None
        self.__dict__.update(kwargs)

    def save(self, f):
        np.save(f, np.asarray(self))

    def __array__(self):
        a = np.empty(1, dtype=self._get_dtype())
        names = self._argnames + self._kwnames
        conversion_lookup = dict()
        cls = self.__class__
        klass_string = '.'.join( (cls.__module__, cls.__name__) )
        for n in names:
            attr = getattr(self, n, None)
            # name rule for obj->array converter is "_array_from_"+n
            arr_composer = '_array_from_'+n
            obj_composer = '_reconstruct_'+n
            
            # handle special array conversion rule first
            if hasattr(self, arr_composer) and hasattr(self, obj_composer):
                conversion_lookup[n] = obj_composer
                a[n][0] = getattr(self, arr_composer)(attr)
                continue

            # in recovering this element, do
            # nothing if either None, or ndarray
            if attr is None or issubclass(type(attr), np.ndarray):
                conversion_lookup[n] = None
            else:
                conversion_lookup[n] = type(attr)

            if isinstance(attr, (list, tuple)):
                attr = np.array(attr, dtype=object)
            
            # this reads weird.. basically avoid doing asarray(None) here
            a[n][0] = np.asanyarray(attr) if attr is not None else None
        a['conversion_lookup'][0] = conversion_lookup
        a['klass_string'][0] = klass_string
        return a

    @staticmethod
    def load(f):
        try:
            arr = np.load(f)
        except AttributeError:
            arr = f
        klass_string = arr['klass_string'][0]
        mod, obj = klass_string.rsplit('.', 1)
        mod = __import__(mod, globals(), locals(), (obj,))
        klass = getattr(mod, obj)
        return klass.from_array(arr)

    @classmethod
    def from_array(klass, a, **kwargs):
        if a.dtype != klass._get_dtype():
            raise ValueError('dtype of input array does not match')

        converters = dict(a['conversion_lookup'][0])

        def convert(tp, arr_obj):
            if tp is None:
##                 print 'not converting', arr_obj
                return arr_obj
            # handle special converter rule first
            if issubclass(type(tp), str):
                converter = getattr(klass, tp)
                return converter(arr_obj)
            if issubclass(tp, array_pickler_mixin):
##                 print 'converting to array_pickler_mixin', arr_obj
                return tp.from_array(arr_obj)
##             print 'converting to some type:', tp, arr_obj
##             if issubclass(tp, (list, tuple)):
            return tp(arr_obj)

        args = (convert(converters[n], a[n][0]) for n in klass._argnames)
        kws = dict( ((n, convert(converters[n], a[n][0]))
                     for n in klass._kwnames) )
        
        # this way, the user-defined keywords take precedence of whatever is
        # found (or not found) in the record array
        kws.update(kwargs)
        return klass(*args, **kws)


def tablify_list(l, nr, nc):
    "Make a 2D table from a 1D sequence (of possibly heterogenous elements)"
    table = []
    if len(l) == nr:
        try:
            if len(l[0]) == nc:
                # Then it's already a table!
                return l
        except TypeError:
            pass
    else:
        assert len(l) == nr*nc, 'There are not enough elements in l to fit '\
               'into a %d x %d table'%(nr, nc)
    while l:
        row = []
        c = 0
        while c < nc:
            row.append(l.pop(0))
            c += 1
        table.append(row)
    return table

