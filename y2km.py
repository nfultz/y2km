"""Main module."""

import numpy
import pandas


# See also https://pandas.pydata.org/pandas-docs/stable/development/extending.html

## TODO port to NDArrayBackedExtensionArray ? https://github.com/pandas-dev/pandas/pull/33660 



## Part 1 - define dtype


@pandas.api.extensions.register_extension_dtype
class Y2kmDtype(pandas.api.extensions.ExtensionDtype):
    """Months since Y2K

    Epoch is set to  0 = Jan 2000

    """
    name = "y2km"
    type = int
    kind = 'i'

    _is_boolean = False
    _is_numeric = True

    @classmethod
    def construct_array_type(cls, *args):
        return Y2kmArray    



## Part 2 - define array


class Y2kmArray(pandas.api.extensions.ExtensionArray):

    dtype = Y2kmDtype

    def __init__(self, values):
        if numpy.isscalar(values):
            values = [values]
        self._m = numpy.array(values, numpy.int16)


    ### Dimensions

    def __getitem__(self, key):
        newValues = self._m[key]

        if type(key) is int:
            return newValues

        return Y2kmArray(newValues)


    def __len__(self):
        return len(self._m)    

    def nbytes(self):
        return self._m.nbytes + 32


    def isna(self):
        return numpy.isnan(self._m)

    def take(self, indices, allow_fill=False, fill_value=None):
        newValues = pandas.core.algorithms.take(self._m, indices, allow_fill=allow_fill, fill_value=fill_value)
        if numpy.isscalar(newValues):
            newValues = [newValues]
        return Y2kmArray(newValues)

    def copy(self):
        return Y2kmArray(self._m.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        to_concat = [x._m for x in to_concat]

        return Y2kmArray(numpy.concatenate(to_concat))

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if scalars[0].__class__ is str:
            return cls._from_sequence_of_strings(scalars)
        return cls(scalars)

    def _formatter(self, boxed=False):
        return lambda x: f"{x // 12 + 2000}-{x % 12 + 1:02d}"

    @staticmethod
    def _string_to_y2km(string):
        yyyy, mm, *_ = string.split("-")
        return (int(yyyy) - 2000)*12 + int(mm) - 1

    @classmethod
    def _from_sequence_of_strings(cls, strings, dtype=None, copy=False):
        return cls(
            list(map(cls._string_to_y2km, strings))
        )

    def astype(self, dtype, copy=True):
        if dtype is str:
            return numpy.array(
                list(map(self._formatter(), self))
            )
        else:
            return super().astype(dtype)

    ## Comparison ops

    def __eq__(self, right):
        if type(right) is str:
            right = Y2kmArray._from_sequence_of_strings([right])

        return self._m == right

    def __lt__(self, right):
        if type(right) is str:
            right = Y2kmArray._from_sequence_of_strings([right])

        return self._m < right._m

    def __le__(self, right):
        return (self < right) | (self == right)

    def __gt__(self, right):
        return ~(self <= right)

    def __ge__(self, right):
        return ~(self < right)

    ## Math ops

    ## Return Y2kms when possible

    def __sub__(self, right):
        if(getattr(right, 'dtype', None) == self.dtype):
            return self._m - right._m

        return Y2kmArray(self._m - right)


    def __add__(self, right):
        if(getattr(right, 'dtype', None) == self.dtype):
            raise Exception()

        return Y2kmArray(self._m + right)

