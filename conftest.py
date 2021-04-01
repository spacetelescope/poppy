# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}

try:
    from poppy import __version__
except ImportError:
    __version__ = ''

PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
PYTEST_HEADER_MODULES['synphot'] = 'synphot'

TESTED_VERSIONS['poppy'] = __version__

## Uncomment the following line to treat all DeprecationWarnings as
## exceptions
# from astropy.tests.helper import enable_deprecations_as_exceptions
# enable_deprecations_as_exceptions()
