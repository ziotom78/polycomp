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
            print "Cache hit: {0}".format((x, y))
            return self.cache[(x, y)]
        else:
            return self.add_point(x, y)

################################################################################

def _metropolis (cost, temperature):
    """Return the Metropolis probability associated with the given `cost' and
    `temperature'.

    See T.J.P. Penna, "Traveling salesman problem and Tsallis statistics",
    Physical Review E, January 1995."""

    number = np.random.uniform (0.0, 1.0)
    return cost < 0.0 or number < np.exp(-cost / temperature)

################################################################################

def find_best_polycomp_parameters(samples, num_of_coefficients_range,
                                  samples_per_chunk_range, max_error,
                                  algorithm, period=None):

    """Performs an optimized search of the best configuration in the
    parameter space given by "num_of_coefficients_space" and
    "samples_per_chunk_space"."""

    best_size = None
    best_parameter_point = None
    best_chunks = None

    optimization_start_time = time.clock()

    init_phase = True
    x_range = num_of_coefficients_range
    y_range = samples_per_chunk_range

    midpoint_x, midpoint_y = [int(np.mean(k)) for k in (x_range, y_range)]

    temperature = 0.5
    cooling_factor = 0.9
    cooling_speed = 10

    param_points = PointCache(samples=samples,
                              max_allowable_error=max_error,
                              algorithm=algorithm,
                              period=period)
    chunks, point = param_points.get_point(midpoint_x, midpoint_y)
    print('Start at point {0}, {1}'.format(midpoint_x, midpoint_y))

    iteration = 0
    while True:
        successful_mutations = 0
        for k in range(cooling_speed):
            mut_x = np.random.random_integers(x_range[0], x_range[1])
            mut_y = np.random.random_integers(midpoint_y - 10, midpoint_y + 10)

            if mut_x < x_range[0] or mut_x > x_range[1]:
                continue
            if mut_y < y_range[0] or mut_y > y_range[1]:
                continue
            if (mut_x, mut_y) == (midpoint_x, midpoint_y):
                continue

            mut_chunks, mut_point = param_points.get_point(mut_x, mut_y)
            print('I am at point {0}, {1} ({2})'.format(mut_x, mut_y,
                                                        mut_point.compr_data_size))

            if _metropolis(mut_point.compr_data_size - point.compr_data_size,
                           temperature):
                chunks = mut_chunks
                point = mut_point
                successful_mutations += 1
                midpoint_x = mut_x
                midpoint_y = mut_y
                print('Mutation {0} accepted, size is now {1}'
                      .format((midpoint_x, midpoint_y), point.compr_data_size))

            iteration += 1

            if successful_mutations >= 5:
                break

        temperature *= cooling_factor

    if best_chunks is None:
        log.error('polynomial compression parameters expected for table "%s"',
                  table)
        sys.exit(1)

    return best_parameter_point, errors_in_param_space
