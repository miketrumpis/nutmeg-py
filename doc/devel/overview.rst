.. _overview:

========
Overview
========

Nutmeg-Py is a transitional project to research and establish the
porting of NUTMEG tools in Matlab into a Python software library. 

The three principles goals of this phase of the project are to design:

* the data representation for various source and analysis data,
  including MEG reconstructions analyzed on a time-frequency grid
  ("tfbeam" data)
* a non-parametric statistical testing package, to make hypothesis
  testing on this data
* a visualization platform which both emulates the SPM-like NUTMEG
  viewer, and also takes advantage of more modern 3D
  graphics technology.

General Notes And Caveats
=========================

This project is the fruit of roughly six months of work, and is
therefore limited in scope and highly experimental. Read otherwise,
you may say it is incomplete and rough around the edges.

The programming methodology I have tried to follow includes:

* good coverage with docstrings 
* documentation using Sphinx_ --among other advantages, this allows the
  docstrings to be automatically translating into the API
  documentation
* illustration and examples through tests (both doctests and unit
  tests [found in various "tests" directories])
* hierarchical design of classes concerning data objects and analysis
  patterns
