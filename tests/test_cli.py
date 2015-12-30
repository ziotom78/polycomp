#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
import os
import tempfile
import subprocess
import sys

try:
    import astropy.io.fits as fits
except ImportError:
    import pyfits as fits

def check_fits_file(file_name, hdu_dict, hdu_num=1):
    with fits.open(file_name) as f:
        assert len(f[hdu_num].columns) == len(hdu_dict), \
          ('there are {0} columns in the FITS file, but only {1} were expected'.
           format(len(f[hdu_num].columns), len(hdu_dict)))
        for col in f[hdu_num].columns:
            assert col.name in hdu_dict, \
              ('column "{0}" not present in the FITS file'.format(col.name))
            array = f[hdu_num].data.field(col.name)
            assert np.allclose(array, hdu_dict[col.name]), \
              ('the data in column "{colname}" of HDU {hdunum} ({hdudata}) differ from {expected}'
               .format(colname=col.name,
                       hdunum=hdu_num,
                       hdudata=array,
                       expected=hdu_dict[col.name]))

def call_polycomp(args, expected_result=0):
    with tempfile.TemporaryFile(mode='w+') as stdout_file:
        with tempfile.TemporaryFile(mode='w+') as stderr_file:
            result = subprocess.call([sys.executable, './polycomp.py'] + args,
                                     stdout=stdout_file,
                                     stderr=stderr_file)

            if result != expected_result:
                for name, file_obj in [('STDOUT', stdout_file),
                                       ('STDERR', stderr_file)]:
                    file_obj.seek(0)
                    print('*** {name} ***'.format(name=name))
                    print(''.join(file_obj.readlines()))
                    print('\n')

            assert result == expected_result


def test_help():
    call_polycomp(['--help'])
    call_polycomp(['compress', '--help'])
    call_polycomp(['decompress', '--help'])

def test_version():
    call_polycomp(['--version'])

class TestCompression:
    def setUp(self):
        with tempfile.NamedTemporaryFile(suffix='.conf', delete=False) as conf_file:
            text = bytes('''[polycomp]
tables = A, B

[A]
file = %(fits_test_file)s
hdu = 1
column = A
compression = none
datatype = int32

[B]
file = %(fits_test_file)s
hdu = 1
column = B
compression = none
datatype = float64
'''.format(fits_test_file='tests/test.fits').encode('utf-8'))
            conf_file.write(text)

            self.conf_file_name = conf_file.name

        # Create an empty file: we're just interested in its name
        with tempfile.NamedTemporaryFile(suffix='.pcomp', delete=False) as pcomp_file:
            self.pcomp_file_name = pcomp_file.name

        self.key_value = 'fits_test_file={0}'.format('tests/test.fits')

    def tearDown(self):
        try:
            os.unlink(self.conf_file_name)
            os.unlink(self.pcomp_file_name)
        except:
            pass

    def test_plain_compression(self):
        call_polycomp(['compress',
                       self.conf_file_name,
                       self.pcomp_file_name,
                       self.key_value])
        check_fits_file(self.pcomp_file_name, {'A': np.array([1, 2, 3])}, 1)
        check_fits_file(self.pcomp_file_name, {'B': np.array([1, 2, 3])}, 2)

    def test_one_table(self):
        call_polycomp(['compress', '-t', 'A',
                       self.conf_file_name,
                       self.pcomp_file_name,
                       self.key_value])
        check_fits_file(self.pcomp_file_name,
                        {'A': np.array([1.0, 2.0, 3.0])})

    def test_many_tables(self):
        call_polycomp(['compress', '-t', 'A', '-t', 'B',
                       self.conf_file_name,
                       self.pcomp_file_name,
                       self.key_value])
        check_fits_file(self.pcomp_file_name, {'A': np.array([1, 2, 3])}, 1)
        check_fits_file(self.pcomp_file_name, {'B': np.array([1, 2, 3])}, 2)
