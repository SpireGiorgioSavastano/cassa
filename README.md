# Classification with Automated Semi-Supervised Algorithms (CASSA) package

![test-main](https://github.com/giorgiosavastano/cassa/actions/workflows/python-test-main.yml/badge.svg)
![coverage-main](https://img.shields.io/codecov/c/github/giorgiosavastano/cassa)
![license](https://img.shields.io/github/license/giorgiosavastano/cassa)

## Overview

`CASSA` is a Python package to perform unsupervised and semi-supervised machine learning (ML) classification algorithms on generic tensors of pre-processed data, such as time series, altitude profiles, images, DDMs and spectra. Mainly tested on Earth Information (EI) data produced by our own satellites, such as GNSS-RO sTEC profiles and GNSS-R DDMs. It produces a database of labeled clusters that can be used to classify new unlabeled data.
The documentation is available at <https://cassa.readthedocs.io/en/latest/>.

It includes the following blocks:

* Parallelized distance matrix computation using earth mover's distance (EMD, aka Wasserstein metric)
* Spetral clustering using precomputed distance matrix
* Self-tuned spectral clustering using precomputed distance matrix
* HDBSCAN clustering using precomputed distance matrix
* Classification of new data based on database of labeled clusters

## Installation

    pip install cassa


### Authors

- Giorgio Savastano (<giorgiosavastano@gmail.com>)
- Karl Nordstrom (<karl.am.nordstrom@gmail.com>)

## References

Savastano, G., K. Nordström, and M. J. Angling (2022), Semi-supervised Classification of Lower-Ionospheric Perturbations using GNSS Radio Occultation Observations from Spire’s Cubesat Constellation. Submitt. to JSWSC.
