#!/bin/bash -e

venvname=rusty-machine-toy
pythonexe=/usr/bin/python3.5

(workon | grep $venvname) || mkvirtualenv --python=$pythonexe $venvname
