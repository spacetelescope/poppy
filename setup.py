from distribute_setup import use_setuptools
use_setuptools()
from setuptools import setup


setupargs = {
    'name'          :       'poppy',
    'version'       :      	"0.0.0",  # will be replaced below
    'description'   :       'Physical optics propagation (wavefront diffraction) for optical simulations, particularly of telescopes.',
    'fullname'      :       'Physical Optics Propagation in PYthon (POPPY)',
    'author'        :     	"Marshall Perrin",
    'author_email'  :      	"mperrin@stsci.edu",
    'url'           :  		"http://www.stsci.edu/~mperrin/software/webbpsf",
    'download_url'           :  		"http://www.stsci.edu/~mperrin/software/webbpsf/poppy-0.0.0.tar.gz",  # will be replaced below
    'platforms'     :      	["Linux","Mac OS X", "Win"],
    'requires'      :       ['pyfits','numpy', 'matplotlib', 'scipy', 'asciitable'],
    'packages'      :       ['poppy'],
    'test_suite'    :       'poppy.tests',
    'classifiers'   :   [
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Development Status :: 4 - Beta"
        ],
    'long_description': """

Physical Optics Propagation in PYthon (POPPY)
------------------------------------------------

    This module implements an object-oriented system for modeling physical optics propagation with diffraction, particularly for multiwavelength simulations of telescopic and coronagraphic imaging. 
    
    Right now only far-field diffraction (Fraunhofer regime) between image and pupil planes is supported; Fresnel propagation and intermediate planes are a future goal.

    Developed by Marshall Perrin at STScI, 2010-2012, for use simulating the James Webb Space Telescope. 

    Documentation can be found online at http://www.stsci.edu/jwst/software/webbpsf/


    """
#    'entry_points'  :       {'gui_scripts': ['webbpsfgui = webbpsf.gui',]} # should create exe file on Windows?
#    'data_files'    :     	[ ( pkg+'/data',  ['data/*' ] ),
#                                ( pkg+'/tests', [ 'tests/*'] ),
 #                           ]
    }

#============ read in the version number from the code itself, following  ============
# http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package

# don't actually import the _version.py, for the reasons described on that web page. 
import re
VERSIONFILE="poppy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    setupargs['version'] = mo.group(1)
    setupargs['download_url'] = "http://www.stsci.edu/~mperrin/software/webbpsf/poppy-"+mo.group(1)+".tar.gz"
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))



# Now actually call setup

setup(**setupargs)
