Appendix C: Developer Notes and Release Procedure
=================================================


Release Process
---------------

#. Get the `develop` branch to the state you would like for the release. This includes all tests passing, both on Travis and locally via tox.
#. Locally, checkout the `develop` branch and ensure it's current with respect to Github.
#. Tag the latest commit in `develop` with `v<version>`, being sure to sign the tag with the `-s` option. (You will need to ensure you have git commit signing via GPG setup prior to this).
   * ``git tag -s v<version> -m "Release v<version>"``

#. Push tag to github:  ``git push upstream v<version>``
#. On github, make a PR from `develop` to `stable` (this can be done ahead of time and left open, until all individual PRs are merged into `develop`.).
#. After verifying that PR is complete and waiting for tests to pass, merge it. (Once merged, both the `stable` and `develop` branches should match).
#. Release on Github:

   #. On Github, click on "[N] Releases".
   #. Select "Draft a new release".
   #. We want to create the release from the tag just added. To do so, select 'Tags', then on the line for the latest tag ``v<version>``, at the far right click on the ``...`` button to bring up a small menu containing "Create release".
   #. Specify the version number, title, and brief description of the release.
   #. Press "Publish Release".

#. Release to PyPI. This should now happen automatically on Travis. This is triggered by a Travis build of a tagged commit on the `stable` branch, so should happen automatically after the PR to `stable`.

#. Release to AstroConda, via steps below.

Releasing a new version through AstroConda
------------------------------------------

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
