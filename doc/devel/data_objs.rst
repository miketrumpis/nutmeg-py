
============
Data Objects
============

Before describing the domain-specific data, one architectural decision
concerning data deserves special attention. 

:class:`~nutmeg.utils.array_pickler_mixin`
++++++++++++++++++++++++++++++++++++++++++

The :class:`~nutmeg.utils.array_pickler_mixin` class of objects
essentially can save-itself-to/load-itself-from a file on disk, in the
"npy" format defined for a Numpy array (an ndarray). These two
functions are defined on an *array_pickler_mixin* object through its
methods :meth:`~nutmeg.utils.array_pickler_mixin.save` and
:meth:`~nutmeg.utils.array_pickler_mixin.load`. By inheriting
from this class, many objects throughout Nutmeg-Py gain this
mechanism, at the cost of having a mildly templated definition.  

The saving/loading is accomplished by rolling all of an object's data
into a special, flexible ndarray called a structured array, where each
slot of the array holds a pointer to some Python data type. The
pattern that must be followed in order for a class of objects to also
be an *array_pickler_mixin* is described as follows (from the class
docstring):

1. define two class-level fields:
  
  *  _argnames -- the list of names of positional arguments in the
     constructor 
  *  _kwnames  -- the list of names of keyword arguments in the
     constructor 

2. The positional arguments must be listed in the correct order.
3. The names in both lists must be attributes of the class objects.
4. Each name in the kwnames list must be identical to both the
   corresponding class object attribute AND the keyword argument name in
   the class constructor. 
5. All attributes named in the lists must be one of:
 
  * ndarray
  * array_pickler_mixin subclass
  * any type that can be reconstructed such that 
    a == type(a)(asarray(a)) (this is satisfied by many built-in types)
  * any type that has special staticmethods **'_array_from_{attr}'** and
    **'_reconstruct_{attr}'** defined on the class. The first method
    creates an ndarray from an object of this type, and the second one
    reconstructs an object in a round-trip sense.  

The main idea is that all of the information needed to construct each
object is saved within the object, and enables the object to be
recreated later (from data off the disk). Thus, the
*array_pickler_mixin* is intended to be used to represent objects that
are essentially a list of data and metadata. 

The construction signature, then, must be along the lines of::

  >>> def __init__(self, data1, data2, info1, option1=True, option2='auto'):

Additionally, each of these positional and keyword arguments must be
then defined as attributes on the object (although this will happen
automatically, as described in a moment). 

To inform the object of its own structure, the class must be defined with
a list of its data and metadata attributes--the **_argnames** and
**_kwnames** class-level lists. Each list has a distinct
constraint. It is important that **_argnames** lists the
attribute-names of the positional arguments *in order*, and that
**_kwnames** lists exactly the same names as the corresponding
attributes. Good examples of these principles can be found in the
class docstring, as well as in the class tests.

The most basic *array_pickler_mixin* simply lists arguments and
keyword arguments. In addition to saving and loading, the naming of
attributes is done automatically, due to a "bucket" design. For
example::

  >>> from nutmeg import utils
  >>> class A(utils.array_pickler_mixin):
  ...     _argnames = ['la', 'di', 'da']
  ... 
  >>> a1 = A(1, 'two', (3,4,5))
  >>> a1.la, a1.di, a1.da
  (1, 'two', (3, 4, 5))

More rich objects, of course, will model the questions being asked
about the data, and will only fall back on this mixin class in order
to save/load the data.

Compatible Data
---------------

Most data types can be saved and loaded without a fuss, as they fall
into the first three categories under item "5" in the list above. For
more complicated objects, an *array_pickler_mixin* may define a pair
of static methods that translate the attribute to/from an ndarray. For
example, the :class:`~nutmeg.core.beam.Beam` class must translate its
*coordmap* attribute, which is an object defined from the NIPY
library. You can see that the class has specially named static
functions that convert coordinate mapping information through the two
methods :meth:`~nutmeg.utils.parameterize_cmap` and :meth:`~nutmeg.utils.cmap_from_array`.

MEG Data
++++++++

The first challenge of the Nutmeg-Py project was to translate the
existing Matlab based data representation into a Python class.




