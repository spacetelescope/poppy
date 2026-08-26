Appendix C: Developer Notes and Release Procedure
=================================================


Release Process
---------------

Preparatory
~~~~~~~~~~~

#. Create a clean conda environment with the current poppy python version
#. Install poppy (including docs) with `pip install -e .[all,test,docs]`. If you use `zsh`
   you will need to escape the brackets, as `.\\[all,test,docs\\]`
#. Install `pandoc` (https://pandoc.org/installing.html)
#. Install `graphviz` (on a mac, with homebrew, `brew install graphviz`)

Creating the Release
~~~~~~~~~~~~~~~~~~~~

#. Get the `develop` branch to the state you would like for the release. This includes all 
   tests passing, both on Github Actions and locally via tox.
#. Locally, checkout the `develop` branch and ensure it's current with respect to Github.
#. On github, create a draft release with target "develop", generate release notes, and
   copy those release notes into the "relnotes.rst" file in docs (reformatting them into
   proper ReStructuredText).
#. Locally, move to the docs directory and type `make html`. Create a branch to fix any
   warnings or errors, and make a PR to bring those fixes into `develop`
#. Locally, check out the `test_readthedocs` branch, update it from develop, and make sure
   that the generated documentation on readthedocs looks correct.
   
   #. `git checkout test_readthedocs`
   #. `git reset --hard develop`
   #. `git push origin test_readthedocs --force` (may be `upstream` instead of `origin`)

#. Locally, test uploading poppy to test pypi

    #. Create and activate a new empty environment
    #. Install poppy (develop branch) into that environment
    #. `pip install build twine`
    #. `python -m build`
    #. `twine check dist/*`
    #. `twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose`
    #. Create and activate a new empty environment
    #. `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple poppy==VERSION`

#. Tag the latest commit in `develop` with `v<version>`, being sure to sign the tag with 
   the `-s` option. (You will need to ensure you have git commit signing via GPG setup 
   prior to this).

   * ``git tag -s v<version> -m "Release v<version>"``

#. Push tag to github:  ``git push upstream v<version>`` (may be `origin` instead of 
   `upstream`)
#. On github, make a PR from `develop` to `stable` (this can be done ahead of time and 
   left open, until all individual PRs are merged into `develop`.).
#. After verifying that PR is complete and waiting for tests to pass, merge it. (Once 
   merged, both the `stable` and `develop` branches should match).
#. Release on Github:

   #. On Github, click on "[N] Releases".
   #. Select "Draft a new release".
   #. We want to create the release from the tag just added. To do so, select 'Tags', 
      then on the line for the latest tag ``v<version>``, at the far right click on the 
      ``...`` button to bring up a small menu containing "Create release".
   #. Specify the version number, title, and brief description of the release.
   #. Press "Publish Release".

#. Release to PyPI. This should now happen automatically on Github Actions. This is 
   triggered by a Github Actions build of a tagged commit on the `stable` branch, so 
   should happen automatically after the PR to `stable`.
#. Test by creating an empty environment, and installing the new poppy version to this
   environment using `pip install poppy==VERSION`
#. Release to AstroConda, via steps below. (Currently deprecated, awaiting transfer to 
   Stenv to update this process)

Releasing a new version through AstroConda
------------------------------------------

.. admonition:: **Consider this section deprecated as of version 1.0.3!**

 AstroConda is currently limited to Python <=3.7, while POPPY only supports Python >=3.8 as of v1.0.3.

Do this after you've done the above.

To test that an Astroconda package builds, you will need ``conda-build``::

   $ conda install conda-build

#. Fork (if needed) and clone https://github.com/astroconda/astroconda-contrib
#. Edit `poppy/meta.yaml <https://github.com/astroconda/astroconda-contrib/blob/master/poppy/meta.yaml>`_ to reflect the new ``version`` and ``git_tag``.
#. Edit in the ``git_tag`` name from ``git tag`` in the PyPI release instructions (``v0.X.Y``).
#. Verify that you can build the package from the astroconda-contrib directory: ``conda build -c http://ssb.stsci.edu/astroconda poppy``
#. Commit your changes to a new branch and push to GitHub.
#. Create a pull request against ``astroconda/astroconda-contrib``.
#. Wait for SSB to build the conda packages.
#. (optional) Create a new conda environment to test the package installation.
