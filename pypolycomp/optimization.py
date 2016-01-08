# -*- encoding: utf-8 -*-

from collections import namedtuple, OrderedDict
import numpy as np
import time

import pypolycomp._bindings as ppc

ParameterPoint = namedtuple('ParameterPoint',
                            ['chunks',
                             'params',
                             'compr_data_size',
                             'num_of_chunks',
                             'num_of_compr_chunks',
                             'num_of_cheby_coeffs',
                             'cr',
                             'elapsed_time'])

################################################################################

def numpy_array_size(arr):
    return arr.itemsize * arr.size

################################################################################

def sample_polycomp_configuration(samples, params):

    start_time = time.clock()
    chunks = ppc.compress_polycomp(samples, params)
    end_time = time.clock()

    chunk_bytes = chunks.num_of_bytes()

    cur_point = ParameterPoint(chunks=chunks,
                               params=params,
                               compr_data_size=chunk_bytes,
                               num_of_chunks=len(chunks),
                               num_of_compr_chunks=chunks.num_of_compressed_chunks(),
                               num_of_cheby_coeffs=chunks.total_num_of_cheby_coeffs(),
                               cr=numpy_array_size(samples) / float(chunk_bytes),
                               elapsed_time=end_time - start_time)

    return chunks, cur_point

################################################################################

class PointCache:
    def __init__(self,
                 samples,
                 max_allowable_error,
                 algorithm,
                 period,
                 num_of_elements_in_cache=15):

        self.samples = samples
        self.max_allowable_error = max_allowable_error
        self.algorithm = algorithm
        self.period = period
        self.num_of_elements = num_of_elements_in_cache
        self.cache = OrderedDict()

        self.best_point = None
        self.best_size = None

    def compute_point(self, x, y):
        params = ppc.Polycomp(num_of_coeffs=x,
                              num_of_samples=y,
                              max_allowable_error=self.max_allowable_error,
                              algorithm=self.algorithm)
        if self.period is not None:
            params.set_period(self.period)

        chunks, point = sample_polycomp_configuration(self.samples, params)
        if self.best_size is None or point.compr_data_size < self.best_size:
            self.best_size = point.compr_data_size
            self.best_point = (chunks, point)

        return chunks, point

    def add_point(self, x, y):
        # If it is already present, delete it so that it will be
        # readded at the end of the list (with higher priority)
        if (x, y) in self.cache:
            new_point = self.cache[(x, y)]
            del self.cache[(x, y)]
        else:
            new_point = self.compute_point(x, y)

        if len(self.cache) >= self.num_of_elements:
            del self.cache[self.cache.keys()[0]]

        self.cache[(x, y)] = new_point
        return new_point

    def get_point(self, x, y):
        if (x, y) in self.cache:
            return self.cache[(x, y)]
        else:
            return self.add_point(x, y)

################################################################################

def find_best_polycomp_parameters(samples, num_of_coefficients_range,
                                  samples_per_chunk_range, max_error,
                                  algorithm, delta_coeffs=1, delta_samples=1,
                                  period=None, callback=None):

    """Performs an optimized search of the best configuration in the
    parameter space given by "num_of_coefficients_space" and
    "samples_per_chunk_space"."""

    optimization_start_time = time.clock()

    x_range = num_of_coefficients_range
    y_range = samples_per_chunk_range

    midpoint_x, midpoint_y = [int(np.mean(k)) for k in (x_range, y_range)]
    param_points = PointCache(samples=samples,
                              max_allowable_error=max_error,
                              algorithm=algorithm,
                              period=period)

    # The logic of this code is the following:
    #
    # 1. Start from a point (x, y)
    # 2. Sample the point and all its neighbours
    # 3. Move to the best point among the nine that have been sampled
    # 4. Repeat from point 2. until the best point is the current one
    #
    # Many points will be sampled more than once, but we use a
    # "PointCache" object to do all the sampling, so that only newer
    # points need to be recalculated every time.

    errors_in_param_space = {}
    num_of_steps = 1
    dx = delta_coeffs
    dy = delta_samples
    while True:
        ring_of_points = [(-dx, -dy), (0, -dy), (dx, -dy),
                          (-dx,   0), (0,   0), (dx,   0),
                          (-dx,  dy), (0,  dy), (dx,  dy)]

        ring_of_configurations = []
        for dx, dy in ring_of_points:
            cur_x = midpoint_x + dx
            cur_y = midpoint_y + dy
            if cur_x < x_range[0] or cur_x > x_range[1]:
                continue
            if cur_y < y_range[0] or cur_y > y_range[1]:
                continue

            chunks, params = param_points.get_point(cur_x, cur_y)
            if callback is not None:
                callback(cur_x, cur_y, params, num_of_steps)

            errors_in_param_space[(cur_x, cur_y)] = params
            ring_of_configurations.append((cur_x, cur_y, chunks, params))

        ring_of_configurations.sort(key=lambda p: p[3].compr_data_size)
        best_x, best_y, best_chunks, best_params = ring_of_configurations[0]
        if (best_x, best_y) == (midpoint_x, midpoint_y):
            break

        midpoint_x, midpoint_y = best_x, best_y
        num_of_steps += 1

    return best_params, errors_in_param_space.values(), num_of_steps
