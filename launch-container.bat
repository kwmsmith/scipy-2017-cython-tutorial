REM 2017-06-26 Untested; please provide feedback!


docker run -it --rm -p 8888:8888 -v %cd%:/home/jovyan/work --name cython-tutorial jupyter/scipy-notebook
