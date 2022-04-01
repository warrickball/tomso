---
title: 'tomso: TOols for Models of Stars and their Oscillations'
tags:
  - Python
  - astronomy
authors:
  - name: Warrick H. Ball
    orcid: 0000-0002-4773-1017
    affiliation: 1
affiliations:
 - name: School of Physics & Astronomy, University of Birmingham, United Kingdom
   index: 1
date: 01 April 2022
bibliography: paper.bib

---

# Summary

Many branches of astronomy interpret observations through approximate,
one-dimensional models of stars.  Furthermore, many stars are observed
to undergo resonant pulsations, and the frequencies at which they do
so are similarly interpreted through the predicted pulsations of the
one-dimensional models.  Since the problem was defined in its modern
form around the mid-20th century, many programs have been written to
produce models and predict their frequencies and correspondingly many
custom data formats have been defined, with varying levels of ease-of-use
and interoperability.  `tomso`'s main purpose is to provide a high-level
interface for parsing data in these formats and simplify research that
uses them.

# Statement of need

Data formats for stellar models and their oscillations vary widely.
Some are fixed-width plain text files with several blocks; some are
Fortran binaries; and few can easily be read with standard routines
for loading data.  Some programs require data to be prepared in a
specific format and provide tools to do so but these tools are
incomplete and difficult to extend.

`tomso` is a set of Python modules that provides a high-level
interface to several formats and enables or simplifies several common
tasks in analysing the stellar models and oscillations.  First, it
allows the user to load the data for inspection, which can be
cumbersome, given the complicated specifications of some formats.  The
high-level interface also allows straightforward access to complicated
properties that can be derived from the data.  Second, it can be
convenient—and, as mentioned above, is sometimes necessary—to convert
files between different formats.  Finally, some computational
experiments involve modifying the stellar models directly, which
requires reading the file in the correct format, manipulating the
data, then ensuring that the data is correctly re-written in the same
format.

For example, two derived properties are the speed of sound
$c_\mathrm{s}$ and acoustic depth $\tau$, which is how long a sound
wave would take to travel from a star's surface to some depth inside.
The sound speed gradient $dc_\mathrm{s}/d\tau$ is useful for
identifying regions where sudden changes in the star's structure might
discernibly affect the frequencies at which it vibrates.
Such a figure is shown for a standard solar model, Model S [@modelS],
in the textbook by @acdk2010 [their Fig. 7.30, which is in essence the
same as Fig. 1 in @monteiro2000] but requires some manipulation of the
source data because neither $c_\mathrm{s}$ nor $\tau$ are part of the
data format.  Fig. \autoref{fig:dc_dtau} shows the same data but with
`tomso` this is naturally expressed in two functional lines of Python code: one
to read the data file and one to plot the relevant data using the
high-level properties `cs` and `tau`.

`tomso`'s interfaces also aim to be easily extensible, so that in the
future it can not only support more current codes and file formats,
but also those that have yet to be developed.

![Plot of the sound speed gradient as a function of the acoustic depth
in a standard solar model [Model S, @modelS], as in Fig. 7.30 of
@acdk2010.\label{fig:dc_dtau}](https://tomso.readthedocs.io/en/stable/index-1.hires.png)

# Acknowledgements

This work has indirectly been supported by the Deutsche
Forschungsgemeinschaft (DFG) through grant SFB 963/1 ``Astrophysical
Flow Instabilities and Turbulence'' (Project A18) and the UK Science
and Technologies Facilities Council (STFC) through grant
ST/R0023297/1.

# References