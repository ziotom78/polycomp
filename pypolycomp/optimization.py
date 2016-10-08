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

class ParameterSpaceMesh:
    def __init__(self,
                 samples_per_chunk,
                 num_of_poly_coeffs,
                 compr_data_size,
                 num_of_chunks,
                 num_of_compr_chunks,
                 num_of_cheby_coeffs,
                 cr,
                 elapsed_time):
        self.samples_per_chunk = samples_per_chunk
        self.num_of_poly_coeffs = num_of_poly_coeffs
        self.compr_data_size = compr_data_size
        self.num_of_chunks = num_of_chunks
        self.num_of_compr_chunks = num_of_compr_chunks
        self.num_of_cheby_coeffs = num_of_cheby_coeffs
        self.cr = cr
        self.elapsed_time = elapsed_time

################################################################################

def copy_param_point_without_chunks(parameter_point):
    import logging as log
    from sys import getsizeof

    newcopy = ParameterPoint(chunks=None,
                             params=parameter_point.params,
                             compr_data_size=parameter_point.compr_data_size,
                             num_of_chunks=parameter_point.num_of_chunks,
                             num_of_compr_chunks=parameter_point.num_of_compr_chunks,
                             num_of_cheby_coeffs=parameter_point.num_of_cheby_coeffs,
                             cr=parameter_point.cr,
                             elapsed_time=parameter_point.elapsed_time)

    return newcopy

################################################################################

def get_param_space_mesh(parameter_point_list):
    '''Return a mesh grid containing'''

    # Remove duplicates and sort the values along the X and Y axes
    samples_per_chunk = np.array(sorted(set([x.params.samples_per_chunk()
                                             for x in parameter_point_list])))
    num_of_poly_coeffs = np.array(sorted(set([x.params.num_of_poly_coeffs()
                                              for x in parameter_point_list])))

    # We assign this to two short-named variables `x` and `y` because
    # we're going to use them quite often below
    x, y = np.meshgrid(samples_per_chunk, num_of_poly_coeffs)

    # At the beginning, the parameter space is zero everywhere. We're
    # going to fill it in the `for` loop below
    mesh = ParameterSpaceMesh(samples_per_chunk=x,
                              num_of_poly_coeffs=y,
                              compr_data_size=np.zeros(x.shape, dtype='int'),
                              num_of_chunks=np.zeros(x.shape, dtype='int'),
                              num_of_compr_chunks=np.zeros(x.shape, dtype='int'),
                              num_of_cheby_coeffs=np.zeros(x.shape, dtype='int'),
                              cr=np.zeros(x.shape, dtype='float'),
                              elapsed_time=np.zeros(x.shape, dtype='float'))

    import logging as log
    for cur_point in parameter_point_list:
        # Find the position of the current point in the mesh grid
        idx_x = np.clip(np.searchsorted(x[0], cur_point.params.samples_per_chunk()),
                        0, len(x[0]) - 1)
        idx_y = np.clip(np.searchsorted(y[:,0], cur_point.params.num_of_poly_coeffs()),
                        0, len(y[0]) - 1)

        # Copy the parameters of the current point in each of the 2D
        # matrices in `mesh`
        for param in ('compr_data_size',
                      'num_of_chunks',
                      'num_of_compr_chunks',
                      'num_of_cheby_coeffs',
                      'cr',
                      'elapsed_time'):
            mesh.__dict__[param][idx_y, idx_x] = getattr(cur_point, param)

    return mesh

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
        # This dictionary is going to keep `self.num_of_elements` items at most
        self.cache = OrderedDict()

        # This is going to contain all elements in `self.cache`, but
        # its size has no upper boundary. The `ParameterPoint.chunks`
        # however is equal to None, in order to save memory. This
        # field is used to do a post-mortem analysis of the
        # convergence.
        self.parameter_space = {}

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
            new_point = self.cache.pop((x, y))
        else:
            new_point = self.compute_point(x, y)

        if len(self.cache) >= self.num_of_elements:
            self.cache.popitem(last=False)

        self.cache[(x, y)] = new_point
        self.parameter_space[(x, y)] = copy_param_point_without_chunks(new_point[1])
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
                                  period=None, callback=None, max_iterations=0):

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

            ring_of_configurations.append((cur_x, cur_y, chunks, params))

        ring_of_configurations.sort(key=lambda p: p[3].compr_data_size)
        best_x, best_y, best_chunks, best_params = ring_of_configurations[0]

        # If we have ran too much iterations, stop bothering and exit the loop
        num_of_steps += 1
        if (max_iterations > 0) and num_of_steps > max_iterations:
            break

        # If we're centered on the best value, let's explore a
        # narrower space around it
        if (best_x, best_y) == (midpoint_x, midpoint_y):
            repeat = False
            # Can the ring be shrunk any further? If so, shrink it and
            # keep iterating
            if (dx > 1) or (dy > 1):
                # If dx == dy, we prefer to reduce dy first
                if dy > dx:
                    dy = dy // 2
                else:
                    dx = dx // 2

                repeat = True

            if repeat:
                continue
            else:
                break

        midpoint_x, midpoint_y = best_x, best_y

    return (best_params,
            list(param_points.parameter_space.values()),
            num_of_steps)


################################################################################

SimplexVertex = namedtuple('SimplexVertex', 'x y value')

def new_vertex(point_cache, coords, callback):
    chunks, params = point_cache.get_point(*coords)
    if callback is not None:
        callback(coords[0], coords[1], params, 0)

    return SimplexVertex(coords[0], coords[1], params.compr_data_size)

def vertex_point(vertex):
    'Return a 2D point suitable for operations with NumPy operators'
    return np.array([int(np.round(vertex.x)), int(np.round(vertex.y))], dtype='float')

def simplex_downhill(samples, num_of_coefficients_range,
                     samples_per_chunk_range, max_error,
                     algorithm, delta_coeffs=1, delta_samples=1,
                     period=None, callback=None, max_iterations=0):

    """Performs an optimized search of the best configuration in the
    parameter space given by "num_of_coefficients_space" and
    "samples_per_chunk_space" using the Nelder-Mead algorithm (AKA downhill
    simplex algorithm)."""

    optimization_start_time = time.clock()

    x_range = num_of_coefficients_range
    y_range = samples_per_chunk_range

    midpoint_x, midpoint_y = [int(np.mean(k)) for k in (x_range, y_range)]
    param_points = PointCache(samples=samples,
                              max_allowable_error=max_error,
                              algorithm=algorithm,
                              period=period)

    num_of_steps = 1
    dx = delta_coeffs
    dy = delta_samples
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    vertexes = [new_vertex(param_points, (midpoint_x, midpoint_y), callback),
                new_vertex(param_points, (midpoint_x + dx, midpoint_y), callback),
                new_vertex(param_points, (midpoint_x, midpoint_y + dy), callback)]
    step_num = 1
    while True:
        # 1 - Order
        vertexes.sort(key=lambda x: x.value)
        vertex_points = [vertex_point(v) for v in vertexes]

        # Check if there are two points that are the same. If it is so, quit
        if np.all(vertex_points[0] == vertex_points[1]) or \
           np.all(vertex_points[1] == vertex_points[2]):
           break

        if max_iterations > 0 and step_num > max_iterations:
           break

        step_num += 1
        # 2 - Centroid
        centroid = np.array([(vertexes[0].x + vertexes[1].x) * 0.5,
                             (vertexes[0].y + vertexes[1].y) * 0.5])

        # 3 - Reflection
        reflected_point = centroid + alpha * (centroid - vertex_points[2])
        if reflected_point.x < 1 or reflected_point.y < 1:
            break

        reflected_vertex = new_vertex(param_points, reflected_point, callback)
        if (vertexes[0].value < reflected_vertex.value) and \
          (reflected_vertex.value < vertexes[1].value):
            vertexes[2] = reflected_vertex
            continue

        # 4 - Expansion
        if reflected_vertex.value < vertexes[0].value:
            expanded_point = reflected_point + gamma * (reflected_point - centroid)
            expanded_vertex = new_vertex(param_points, expanded_point, callback)
            if expanded_vertex.value < reflected_vertex.value:
                vertexes[2] = expanded_vertex
                continue
            else:
                vertexes[2] = reflected_vertex

        # 5 - Contraction
        contracted_point = centroid + rho * (vertex_points[2] - centroid)
        if contracted_point.x < 1 or contracted_point.y < 1:
            break
        contracted_vertex = new_vertex(param_points, contracted_point, callback)
        if contracted_vertex.value < vertexes[2].value:
            vertexes[2] = contracted_vertex
            continue

        # 6 - Reduction
        for i in (1, 2):
            new_point = vertex_points[0] + sigma * (vertex_points[i] - vertex_points[0])
            vertexes[i] = new_vertex(param_points, new_point, callback)

    return (param_points.best_point[1],
            list(param_points.parameter_space.values()),
            num_of_steps)
