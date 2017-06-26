# SciPy 2017 Cython Tutorial

2017-06-26 NOTE: the Windows-specific instructions below are untested.  If you
are a Windows-using tutorial-ee, please provide feedback (pull-request,
email the author, tutorial slack channel) and corrections, thank you!

## Setup Instructions

* Clone this repository and `cd` into it:

```
$ git clone git@github.com:kwmsmith/scipy-2017-cython-tutorial.git
$ cd scipy-2017-cython-tutorial
```

* If you've cloned earlier, `git pull` on the `master` branch to ensure you
have the latest updates:

```
$ git pull
```

The tutorial has a few one-time requirements which we describe below.

## One-stop-shop via Jupyter docker containers

The `jupyter` project has convenient self-contained docker containers with
everything we need for this tutorial.  We recommend this method provided you
can install and use a recent version of docker on your platform.   This path
will ensure you have a functional environment that matches the one used by the
instructor.

* [Install Docker](https://www.docker.com/community-edition) for your OS.
* Mac / Linux: Open up a terminal and execute
    $ ./launch-container.sh
* Windows: Open up a powershell and run
    $ launch-container.bat 

This will download an run the [Jupyter scipy
notebook](https://hub.docker.com/r/jupyter/scipy-notebook/) docker image, and
launch a notebook server.

* If successful, jupyter will publish a URI like

```
http://localhost:8888/?token=5f369cf87f26b0a3e4756e4b28bbd9deadbeef
```

* Visit this URI in your browser, you should see the home page of a Jupyter
notebook server, with a list of all files in this repository.

* Open the `test-setup` ipython notebook, and execute all cells.  All should
execute without error or exception.

* In a separate terminal (Mac / Linux) or powershell (Windows) window, navigate
to this directory and run the `docker-test-xtension.sh` command (Mac / Linux)
or the `docker-test-xtension.bat` command (Windows).

## Platform-specific (non-docker) setup instructions

These instructions are for those who can't or don't want to use the recommended
docker-based installation above.

### Mac-specific setup (non-docker)

* Ensure you already have XCode / Mac OS developer tools / command line tools
installed; if not, do so for your version of Mac OS.  Check your install by
running the following from the commandline:

```
$  gcc --version
Configured with: --prefix=/Library/Developer/CommandLineTools/usr --with-gxx-include-dir=/usr/include/c++/4.2.1
Apple LLVM version 8.0.0 (clang-800.0.42.1)
Target: x86_64-apple-darwin15.6.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```

### Linux-specific setup (non-docker)

* Ensure you have the packages necessary for `gcc` and related headers for your
distribution.  Check that you have a recent version of `gcc` installed:

```
$  gcc --version
```

### Windows-specific setup (non-docker)

NOTE: untested -- please provide feedback!

* [Install Visual Studio 2017](https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/), select the *Python development workload* and the Native development tools option.

### General setup after compiler / dev tools are installed (non-docker)

* If you haven't already, [install Miniconda](https://conda.io/miniconda.html).

* Create a `cython-tutorial` environment for this tutorial, using the
`requirements_conda.txt` file provided:
    $ conda create --yes -n cython-tutorial --file ./requirements_conda.txt
    $ source activate cython-tutorial

* Launch a jupyter notebook server
    $ jupyter notebook

* Open the `test-setup` notebook and run all cells.  All should execute without
error or exception.
