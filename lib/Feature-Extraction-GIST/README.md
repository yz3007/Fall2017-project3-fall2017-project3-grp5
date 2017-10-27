## How to reproduce feature.py in your laptop?
# Authour: Siyi Tao

Before your run it, remember to change path to the directory where you put jpg files.

## Pre-requirements
numpy, PIL, gist, csv, FFTW

# Install numpy
$ pip install numpy

# Install FFTW
FFTW download: http://www.fftw.org
Install instruction: http://www.fftw.org/fftw3_doc/Installation-on-Unix.html

$ ./configure --enable-single --enable-shared
$ make
$ sudo make install

# Install gist

Download lear_gist: https://github.com/tuttieee/lear-gist-python

$ sudo python setup.py build_ext
$ python setup.py install

If fftw3f is installed in non-standard path (for example, $HOME/local), use -I and -L options:

$ sudo python setup.py build_ext -I $HOME/local/include -L $HOME/local/lib


