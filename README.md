# SciPy 2017 Cython Tutorial

<img src="https://imgs.xkcd.com/comics/universal_install_script.png" />

2017-06-26 NOTE: the Windows-specific instructions below are untested.  If you
are a Windows-using tutorial-ee, please provide feedback (pull-request,
email the author, tutorial slack channel) and corrections, thank you!

# End goal

We need just a few things set up for this tutorial:
* The contents of this repository.
* A CPython interpreter (Python 3).
* The Cython package (version 0.25) and a few other dependencies (see
  `requirements_conda.txt`).
* A working C / C++ compiler.

It's the last requirement that can be a challenge, depending on your platform /
OS.  The standard GCC / clang compiler that is available on Linux / Mac
(respectively) will work fine.  Windows can be more of a challenge.

In an effort to make things more uniform, we are using a docker container that
bundles everything together except for the contents of this repository.

## Setup Instructions

* Clone this repository and `cd` into it:

```bash
$ git clone git@github.com:kwmsmith/scipy-2017-cython-tutorial.git
$ cd scipy-2017-cython-tutorial
```

* If you've cloned earlier, `git pull` on the `master` branch to ensure you
have the latest updates:

```bash
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

```bash
$ ./launch-container.sh
```

Leave this terminal as-is and do not exit the running docker session.

Verify that your container is running by opening a separate terminal and running

```bash
$ docker ps
CONTAINER ID        IMAGE                    COMMAND                  CREATED             STATUS              PORTS                    NAMES
deadbeef            jupyter/scipy-notebook   "tini -- start-not..."   7 minutes ago       Up 7 minutes        0.0.0.0:8888->8888/tcp   cython-tutorial
```

You should see output like the above, with a different `CONTAINER ID`.
Importantly, you should see `cython-tutorial` under the `NAMES` column.  You
will see more than one row if you have other docker containers running.

* Windows: Open up a powershell and run

```bash
$ launch-container.bat 
```

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

* You should see output like:

```bash
$ ./docker-test-xtension.sh
running build_ext
building 'xtension.foo' extension
gcc -pthread -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/conda/include/python3.5m -c xtension/foo.c -o build/temp.linux-x86_64-3.5/xtension/foo.o
gcc -pthread -shared -L/opt/conda/lib -Wl,-rpath=/opt/conda/lib,--no-as-needed build/temp.linux-x86_64-3.5/xtension/foo.o -L/opt/conda/lib -lpython3.5m -o build/lib.linux-x86_64-3.5/xtension/foo.cpython-35m-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-3.5/xtension/foo.cpython-35m-x86_64-linux-gnu.so -> xtension
***********************************************************
sys.executable: /opt/conda/bin/python
cython version: 0.25.2
xtension module test (31.415926): 31.415926
***********************************************************
```

* If you see an error like `Error: No such container: cython-tutorial`, then
you likely shut down the docker container before running the test.  Re-launch
the container (`./launch-container.sh`) and in a separate terminal run the
`docker-test-xtension.sh` script again.

## Platform-specific (non-docker) setup instructions

These instructions are for those who can't or don't want to use the recommended
docker-based installation above.

### Mac-specific setup (non-docker)

* Ensure you already have XCode / Mac OS developer tools / command line tools
installed; if not, do so for your version of Mac OS.  Check your install by
running the following from the commandline:

```bash
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

```bash
$  gcc --version
```

### Windows-specific setup (non-docker)

NOTE: untested -- please provide feedback!

* [Install Visual Studio 2017](https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/), select the *Python development workload* and the Native development tools option.

### General setup after compiler / dev tools are installed (non-docker)

* If you haven't already, [install Miniconda](https://conda.io/miniconda.html).

* Create a `cython-tutorial` environment for this tutorial, using the
`requirements_conda.txt` file provided:

```bash
$ conda create --yes -n cython-tutorial --file ./requirements_conda.txt
$ source activate cython-tutorial
```

* Launch a jupyter notebook server

```bash
$ jupyter notebook
```

* Open the `test-setup` notebook and run all cells.  All should execute without
error or exception.
