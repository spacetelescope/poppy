#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# --based on setup.py from astropy--

import glob
import os
import sys

import setuptools_bootstrap
from setuptools import setup, find_packages

import astropy
from astropy.setup_helpers import (register_commands, adjust_compiler,
                                   filter_packages, update_package_files,
                                   get_debug_option)
from astropy.version_helpers import get_git_devstr, generate_version_py

PACKAGENAME = 'poppy'
DESCRIPTION = 'Physical optics propagation (wavefront diffraction) for optical simulations, particularly of telescopes.'
LONG_DESCRIPTION = """Physical Optics Propagation in PYthon (POPPY)
------------------------------------------------

This module implements an object-oriented system for modeling physical optics propagation with diffraction, particularly for multiwavelength simulations of telescopic and coronagraphic imaging. 

Right now only far-field diffraction (Fraunhofer regime) between image and pupil planes is supported; Fresnel propagation and intermediate planes are a future goal.

Developed by Marshall Perrin at STScI, 2010-2012, for use simulating the James Webb Space Telescope. 

Documentation can be found online at http://www.stsci.edu/jwst/software/webbpsf/
"""
AUTHOR = 'Marshall Perrin'
AUTHOR_EMAIL = 'mperrin@stsci.edu'
LICENSE = 'BSD'
URL = 'http://www.stsci.edu/~mperrin/software/webbpsf'



# VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = '0.3rc2'

# Indicates if this version is a release version
RELEASE = 'dev' not in VERSION

if not RELEASE:
    VERSION += get_git_devstr(False)

#DOWNLOAD_BASE_URL = 'http://pypi.python.org/packages/source/p/poppy'

# Populate the dict of setup command overrides; this should be done before
# invoking any other functionality from distutils since it can potentially
# modify distutils' behavior.
cmdclassd = register_commands(PACKAGENAME, VERSION, RELEASE)

# Adjust the compiler in case the default on this platform is to use a
# broken one.
adjust_compiler(PACKAGENAME)

# Freeze build information in version.py
generate_version_py(PACKAGENAME, VERSION, RELEASE, get_debug_option())

## 
## 
## setupargs = {
##     'name'          :       PACKAGENAME,
##     'version'       :      	VERSION,
##     'fullname'      :       'Physical Optics Propagation in PYthon (POPPY)',
##     'author'        :     	"Marshall Perrin",
##     'author_email'  :      	"mperrin@stsci.edu",
##     'url'           :  		"http://www.stsci.edu/~mperrin/software/webbpsf",
##     'download_url'           :  		"http://www.stsci.edu/~mperrin/software/webbpsf/poppy-0.0.0.tar.gz",  # will be replaced below
##     'platforms'     :      	["Linux","Mac OS X", "Win"],
##     'requires'      :       ['pyfits','numpy', 'matplotlib', 'scipy', 'asciitable'],
##     'packages'      :       ['poppy'],
##     'test_suite'    :       'poppy.tests',
##     'classifiers'   :   [
##         "Programming Language :: Python",
##         "License :: OSI Approved :: BSD License",
##         "Operating System :: OS Independent",
##         "Intended Audience :: Science/Research",
##         "Topic :: Scientific/Engineering :: Astronomy",
##         'Topic :: Scientific/Engineering :: Physics',
##         "Development Status :: 4 - Beta"
##         ],
##     'long_description': """
## 
## Physical Optics Propagation in PYthon (POPPY)
## ------------------------------------------------
## 
##     This module implements an object-oriented system for modeling physical optics propagation with diffraction, particularly for multiwavelength simulations of telescopic and coronagraphic imaging. 
##     
##     Right now only far-field diffraction (Fraunhofer regime) between image and pupil planes is supported; Fresnel propagation and intermediate planes are a future goal.
## 
##     Developed by Marshall Perrin at STScI, 2010-2012, for use simulating the James Webb Space Telescope. 
## 
##     Documentation can be found online at http://www.stsci.edu/jwst/software/webbpsf/
## 
## 
##     """
## #    'entry_points'  :       {'gui_scripts': ['webbpsfgui = webbpsf.gui',]} # should create exe file on Windows?
## #    'data_files'    :     	[ ( pkg+'/data',  ['data/*' ] ),
## #                                ( pkg+'/tests', [ 'tests/*'] ),
##  #                           ]
##     }
## # Use the find_packages tool to locate all packages and modules
packagenames = filter_packages(find_packages())

# Treat everything in scripts except README.rst as a script to be installed
scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
           if os.path.basename(fname) != 'README.rst']

# Additional C extensions that are not Cython-based should be added here.
extensions = []

# A dictionary to keep track of all package data to install
package_data = {PACKAGENAME: ['data/*']}

# A dictionary to keep track of extra packagedir mappings
package_dirs = {}

# Update extensions, package_data, packagenames and package_dirs from
# any sub-packages that define their own extension modules and package
# data.  See the docstring for setup_helpers.update_package_files for
# more details.
update_package_files(PACKAGENAME, extensions, package_data, packagenames,
                     package_dirs)


setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      packages=packagenames,
      package_data=package_data,
      package_dir=package_dirs,
      ext_modules=extensions,
      scripts=scripts,
      requires=['astropy'],
      install_requires=['astropy'],
      provides=[PACKAGENAME],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      cmdclass=cmdclassd,
      zip_safe=False,
      use_2to3=False
      )
