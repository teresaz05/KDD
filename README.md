# Optimizing Memory Bandwidth for Efficient Approximate Nearest Neighbor Search

This repository contains information on the implementation of the methods described in the attached paper: "Optimizing Memory Bandwidth for Efficient Approximate Nearest Neighbor Search."

## Overview

Approximate Nearest Neighbor (ANN) search is fundamental to high-dimensional retrieval tasks but is often bottlenecked by memory bandwidth rather than computation. Targeting this problem, this project proposes a memory-centric solution that reduces data transfer during ANN search without altering point representations.

The paper implements:
- Progressive distance computation with early rejection to avoid unnecessary full-precision fetches
- Adaptive thresholding for both cosine similarity and Euclidean distance
- Disaggregated in-memory placement of floating-point data for selective precision access
- Bit-wise shuffling and lossless compression to further cut down memory usage

## Code

The codebase includes:
- Logic for computing approximate distances and applying adaptive rejection thresholds
- Methods for compressing reduced-precision data and evaluating compression ratios
- Scripts to reproduce results and plot accuracy vs. bandwidth trade-offs.

## Datasets

The datasets used in our experiments include: SIFT, GLOVE, GIST, FINEWEB, MS MARCO, and DBPEDIA. However, due to size constraints, these datasets are not included in this repository, but they are all publicly available and can be easily found online.
