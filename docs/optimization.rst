Optimizing polynomial compression parameters
============================================

Polycomp implements a number of tools to simplify the choice of the
parameters for polynomial compression. The parameters are
:math:`N_\mathrm{chunk}`, the number of samples per chunk, and
:math:`\deg p(x) + 1`, the number of coefficients in the interpolating
polynomial. Polycomp's strategy to optimize these parameters is to
test a number of configuration and pick the one with the best
compression ratio :math:`C_r`.

Two algorithms are implemented:

- A naive algorithm that scans rectangular regions of the parameter
  space :math:`N_\mathrm{chunk} \times (\deg p(x) + 1)`;
- A simplex-downhill algorithm that hunts for local minima in the
  parameter space.

Scanning rectangular regions of the parameter space
---------------------------------------------------

To find the best values for :math:`N_\mathrm{chunk}` and :math:`(\deg
p(x) + 1)` within a rectangular region of the parameter plane, the
user can specify the values to be checked using the usual parameters
``num_of_coefficients`` and ``samples_per_chunk``. Instead of
specifying only one value, a set of values can be specified using the
following syntax:

- Ranges can be specified using the syntax ``NN-MM`` or ``NN:MM``;
- Intervals can be specified using the syntax ``NN:D:MM``, where ``D``
  is the interval;
- Multiple values and ranges can be concatenated, using commas (``,``)
  as separators.

Simplex-downhill strategy
-------------------------
