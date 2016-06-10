"""
Create a small summary snapshot of a numpy array.

This will detect most changes in the array, but with tolerance to small
differences.
The result is printable, and the printed form can be converted back into a
functional snapshot.

"""
from collections import OrderedDict

import numpy as np

from dithers import (dither8x8_0_1 as DEFAULT_DITHER_KERNEL,
                     balanced_dither as balanced_dither_array_for_shape)
from sample_indices import sample_indices as sample_indices_for_shape

DEFAULT_RELTOL = 1.0e-7
DEFAULT_ABSTOL = 1.0e-7
DEFAULT_MAX_CATEGORIES = 25
DEFAULT_STATS_DECIMATIONS = []

class ArraySnapshot(object):
    default_reltol = DEFAULT_RELTOL
    default_abstol = DEFAULT_ABSTOL
    default_stats_decimations = DEFAULT_STATS_DECIMATIONS
    default_max_categories = DEFAULT_MAX_CATEGORIES
    default_dither_kernel = DEFAULT_DITHER_KERNEL
    _key_property_names = ['shape', 'mask', 'stats_min', 'stats_max', 'stats_mean',
                           'sample_point_indices', 'sample_point_values',
                           'dithered_mean']
    _control_property_names = ['reltol', 'abstol',
                               'stats_decimations', 'max_categories',
                               'dither_kernel']

    def __init__(self,
                 shape_or_array=None,
                 mask=None,
                 stats_min=None, stats_max=None, stats_mean=None,
                 sample_point_indices=None, sample_point_values=None,
                 dithered_mean=None,
                 reltol=None, abstol=None,
                 stats_decimations=None,
                 max_categories=None,
                 dither_kernel=None):
        """Create an array snapshot from components, or from a given array."""
        # Establish the basic control properties.
        # The defaults of these belong to the class
        reltol = reltol or self.default_reltol
        abstol = abstol or self.default_abstol
        stats_decimations = stats_decimations or self.default_stats_decimations
        max_categories = max_categories or self.default_max_categories
        dither_kernel = dither_kernel or self.default_dither_kernel
        self.reltol = reltol
        self.abstol = abstol
        self.stats_decimations = stats_decimations
        self.max_categories = max_categories
        self.dither_kernel = dither_kernel
        if isinstance(shape_or_array, tuple):
            self.shape = shape_or_array
            self.mask = mask
            self.stats_min = stats_min
            self.stats_min = stats_min
            self.stats_mean = stats_mean
            self.sample_point_indices = sample_point_indices
            self.sample_point_values = sample_point_values
            self.dithered_mean = dithered_mean
        else:
            self._take_snapshot(shape_or_array)

    def _take_snapshot(self, array):
        """"Update to match the given array (plain ndarray, *NOT* masked)."""
        self.shape = array.shape
        # Record snapshot of the mask, if we have one
        if not isinstance(array, np.ma.MaskedArray):
            self.mask = None
            self.fill_value = None
        else:
            # Treat the mask as a total array, so we can compare snapshots.
            self.mask = ArraySnapshot(np.ma.getmaskarray(array))
            self.fill_value = array.fill_value
            # Remainder of the snapshot is done on the filled array.
            array = array.filled()
        # Form stats, decimating if enabled
        if not self.stats_decimations:
            # Use whole array
            stats_array = array
        else:
            # Slice array to get a decimated version for stats calculations.
            slices = [slice(None, None, decimate)
                      for decimate in self.stats_decimations]
            slices = slices[:len(array.shape)]
            stats_array = array[slices]
        self.stats_min = np.min(stats_array)
        self.stats_max = np.max(stats_array)
        self.stats_mean = np.mean(stats_array)
        self.sample_point_indices = sample_indices_for_shape(self.shape)
        self.sample_point_values = [array[inds]
                                    for inds in self.sample_point_indices]
        max_treated_as_categories = len(self.sample_point_indices) / 2
        if self.max_categories < max_treated_as_categories:
            max_treated_as_categories = self.max_categories
        self.is_categorical = (len(set(self.sample_point_values)) <
                               max_treated_as_categories)
        if not self.is_categorical:
            self.dithered_mean = 0.0
        else:
            # Calculate and attach a dithered total mean.
            dither = balanced_dither_array_for_shape(
                self.shape, pattern=self.dither_kernel)
            # TODO: or something ...
            dither = (dither - 0.5) * (self.stats_max - self.stats_min)
            total_array = array + dither
            self.dithered_mean = np.mean(total_array)

    @staticmethod
    def from_str(string):
        """Produce an ArraySnapshot from its string printout."""
        return ArraySnapshot()

    def as_dict(self):
        """Return our key properties as an (ordered) dictionary."""
        return OrderedDict([(keyname, getattr(keyname))
                            for keyname in self._key_property_names])

    def copy(self):
        all_prop_names = (self._key_property_names +
                          self._control_property_names)
        all_props = {keyname:getattr(self, keyname)
                     for keyname in all_prop_names}
        del all_props['shape']
        all_props['shape_or_array'] = self.shape
        return self.__class__(**all_props)

    def matches(self, array):
        # Make a copy of ourself, including the control settings.
        other = self.copy()
        # Take a suitable snapshot of the other array.
        other._take_snapshot(array)
        # Compare the snapshots.
        return self == other

    def __eq__(self, other):
        """Check whether array snapshots are the same."""
        # Start with the easy stuff.
        result = (self.shape == other.shape and
                  self.sample_point_indices == other.sample_point_indices and
                  self.mask == other.mask and
                  self.fill_value == other.fill_value and
                  self.stats_decimations == other.stats_decimations and
                  self.dither_kernel == other.dither_kernel)
        # NOTE: different tolerances do *NOT* make a snapshot non-equal
        #  - the lhs object's values are used for the actual comparisons.

        # Compare key values, with tolerance.
        if result:
            result = np.allclose(self.sample_point_values,
                                 other.sample_point_values,
                                 rtol=self.reltol, atol=self.abstol)

        # Compare stats values, with tolerance
        if result:
            result = np.allclose(
                [self.stats_min, self.stats_max, self.stats_mean],
                [other.stats_min, other.stats_max, other.stats_mean],
                rtol=self.reltol, atol=self.abstol)

        # Compare sample points, with tolerance.
        if result:
            result = np.allclose(self.sample_point_values,
                                 other.sample_point_values,
                                 rtol=self.reltol, atol=self.abstol)

        # Compare dithered means, with tolerance.
        if result:
            result = np.allclose([self.dithered_mean],
                                 [other.dithered_mean],
                                 rtol=self.reltol, atol=self.abstol)

        return result

    def __str__(self):
        return str(self.as_dict())

