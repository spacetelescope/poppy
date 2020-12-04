### Contributing Guide

Please open a new issue or new pull request for bugs, feedback, or new features you would like to see. If there is an issue you would like to work on, please leave a comment and we will be happy to assist. New contributions and contributors are very welcome!

See the instructions below for how to contribute to this repository using a forking workflow.

#### Forking Workflow
1. Create a personal fork of the `poppy` repository by visiting its location on GitHub and clicking the `Fork` button.  This will create a copy of the `poppy` repository under your personal GitHub account (hereby referred to as "personal fork").  Note that this only has to be done once.

2. Make a local copy of your personal fork by cloning the repository (e.g. `git clone https://github.com/username/poppy.git`, found by clicking the green "clone or download" button.).  Note that, unless you explicitly delete your clone of the fork, this only has to be done once.

3. Ensure that the personal fork is pointing to the `upstream` `poppy` repository with `git remote add upstream https://github.com/spacetelescope/poppy.git` (or use the SSH version if you have your SSH keys set up).  Note that, unless you explicitly change the remote location of the repository, this only has to be done once.

4. Create a branch off of the `develop` branch on the personal clone to develop software changes on. Branch names should be short but descriptive (e.g. `new-database-table` or `fix-ingest-algorithm`), and not too generic (e.g. `bug-fix`).  Consistent use of hyphens is encouraged.
    1. `git branch <branchname>`
    2. `git checkout <branchname>` - you can use this command to switch back and forth between existing branches.
    3. Perform local software changes using the nominal `git add`/`git commit -m` cycle:
       1. `git status` -  allows you to see which files have changed.
       2. `git add <new or changed files you want to commit>`
       3. `git commit -m 'Explanation of changes you've done with these files'`

5. Push the branch to the GitHub repository for the personal fork with `git push origin <branchname>`.

6. In the `poppy` repository, create a pull request for the recently pushed branch.  You will want to set the base fork pointing to `poppy:develop` and the `head` fork pointing to the branch on your personal fork (i.e. `username:branchname`).  Note that if the branch is still under development, you can use the GitHub "Draft" feature (under the "Reviewers" section) to tag the pull request as a draft. Not until the "Ready for review" button at the bottom of the pull request is explicitly pushed is the pull request 'mergeable'.

7. Assign the pull request a reviewer, selecting a maintainer of the `poppy` repository.  They will review your pull request and either accept the request and merge, or ask for additional changes.

8. Iterate with your reviewer(s) on additional changes if necessary, addressing any comments on your pull request.  If changes are required, you may end up iterating over steps 4.iii and 5 several times while working with your reviewer.

9. Once the pull request has been accepted and merged, you can delete your local branch with `git branch -d <branchname>`.

#### Keeping your fork updated
If you wish to, you can keep a personal fork up-to-date with the `poppy` repository by fetching and rebasing with the `upstream` remote. Do this for both the `master` branch (shown below) and the `develop` branch:
1. `git checkout master`
2. `git fetch upstream master`
3. `git rebase upstream/master`

#### Collaborating on someone else's fork
Users can contribute to another user's personal fork by adding a `remote` that points to their fork and using the nominal forking workflow, e.g.:

1. `git remote add <username> <remote URL>`
2. `git fetch <username>`
3. `git checkout -b <branchname> <username>/<branchname>`
4. Make some changes (i.e. `add/commit` cycle)
5. `git push <username> <branchname>`
