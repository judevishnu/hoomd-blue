from hoomd.filter.filter_ import _ParticleFilter
from hoomd import _hoomd


class _ParticleFilterSetOperations(_ParticleFilter):
    """An abstract class for `ParticleFilters` with set operations.

    Should not be instantiated directly."""

    @property
    def _cpp_cls_name(self):
        """The name of the C++ class in the `_hoomd` module.

        Used for Python class's inheritance.
        """
        raise NotImplementedError

    @property
    def _symmetric(self):
        """Whether the class implements a symmetric set operation.

        Determines behavior of __eq__.
        """
        raise NotImplementedError

    def __init__(self, f, g):
        if f == g:
            raise ValueError("Cannot use same filter for {}"
                             "".format(self.__class__.__name__))
        else:
            self._f = f
            self._g = g
        # Grab the C++ class constructor for the set operation using the class
        # variable _cpp_cls_name
        getattr(_hoomd, self._cpp_cls_name).__init__(self, f, g)

    def __hash__(self):
        return hash(hash(self._f) + hash(self._g))

    def __eq__(self, other):
        if self._symmetric:
            return type(self) == type(other) and \
                (self._f == other._f or self._f == other._g) and \
                (self._g == other._g or self._g == other._f)
        else:
            return type(self) == type(other) and \
                self._f == other._f and self._g == other._g


class SetDifference(_ParticleFilterSetOperations,
                    _hoomd.ParticleFilterSetDifference):
    _cpp_cls_name = 'ParticleFilterSetDifference'
    _symmetric = False


class Union(_ParticleFilterSetOperations, _hoomd.ParticleFilterUnion):
    _cpp_cls_name = 'ParticleFilterUnion'
    _symmetric = True


class Intersection(_ParticleFilterSetOperations,
                   _hoomd.ParticleFilterIntersection):
    _cpp_cls_name = 'ParticleFilterIntersection'
    _symmetric = True
