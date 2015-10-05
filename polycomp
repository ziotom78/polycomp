#!/usr/bin/env python3
# -*- mode: python -*-

"""Compress/decompress FITS files using polynomial compression and other
algorithms.

Usage:
    polycomp compress <schema>
    polycomp decompress [--output=<output_file>] <input_file>
    polycomp optimize <fits_file>
    polycomp info <input_file>
    polycomp (-h | --help)
    polycomp --version

Options:
    -h, --help              Show this help
    --version               Print the version number of the executable
    --output=<output_file>  Specify the name of the output file
"""

from docopt import docopt
import pytoml as toml
import sys
import pypolycomp as ppc

########################################################################

def do_compress(arguments):
    file_name = arguments['<schema>']
    try:
        with open(file_name, 'r') as f:
            schema = toml.load(f)
            print(schema)

    except (OSError, IOError) as e:
        sys.stderr.write("unable to load file \"{0}\": {1}\n"
                         .format(file_name, e.strerror))

    except toml.TomlError as e:
        sys.stderr.write("invalid TOML file \"{0}\": {1}\n"
                         .format(file_name, e.strerror))

########################################################################

def main():
    arguments = docopt(__doc__,
                       version='Polycomp {0}'.format(ppc.__version__))

    if arguments['--version']:
        print(ppc.__version__)
    elif arguments['compress']:
        do_compress(arguments)
    elif arguments['decompress']:
        pass
    elif arguments['optimize']:
        pass
    elif arguments['info']:
        pass
    else:
        print('Sorry, I do not know what to do!')

if __name__ == "__main__":
    main()
