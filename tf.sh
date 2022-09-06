#!/bin/bash

set -e

# docker pull tensorflow/tensorflow:2.9.1 # for CPU tf
docker pull tensorflow/tensorflow:2.9.1-gpu # for GPU tf
docker buildx build --platform linux/amd64 -t tf .
# docker run -u $(id -u):$(id -g) -t -v=${PWD}:/app tf "flake8 ."
# docker run -u $(id -u):$(id -g) -t -v=${PWD}:/app tf "black ."
docker run -u $(id -u):$(id -g) -t -v=${PWD}:/app tf "find -iname *.py  -print | parallel 'a2ps {} -o {}.ps  && ps2pdf {}.ps {}.pdf'"
docker run -u $(id -u):$(id -g) -ti -v=${PWD}:/app tf "${*}"

# Usage:
# $ ./tf.sh <command string>

# Example invocation:
# $ ./tf.sh python linear-example/main.py


# $ ./tf.sh gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=linear_example.pdf linear-example/main.py.pdf linear-example/fit.pdf
