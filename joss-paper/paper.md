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
 - name: University of Birmingham, UK
   index: 1
date: 30 March 2022
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
and interoperability.  Some are fixed-width plain text files; some are
Fortran binaries; some can be converted from one format to another
(but not necessarily back again).

`tomso` is a set of Python modules that provides a high-level
interface to several formats, including the ability to convert between
different file formats, which can be convenient or even
necessary.  It also allows for high-level manipulation of the data
files themselves, through which a user can experiment with the models
and preserve the relevant formatting.  Finally, the high-level
interface allows convenient access to properties that can be derived
from the data.

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
data format.  Fig. \autoref{dc_dtau} shows the same data but with
`tomso` this is naturally expressed in two lines of Python code: one
to read the data file and one to plot the relevant data using the
high-level properties `cs` and `tau`.  ![Easy
figure.\label{dc_dtau}](https://tomso.readthedocs.io/en/stable/index-1.hires.png){
width=100% }

# References
