# FeatureExtraction
Simple Implementation of Information Gain, Mutual Information and Chi-Square Algorithms for Feature Extraction.

## Introduction
 * Information Gain
 * Mutual Information
 * Chi-Square
 * Classify using SVM

A part of Hamshahri Corpus was used as dataset. I tried to extract 100 most important features from different news categories.

## Uses
 - Numpy (http://www.numpy.org/) version 1.14.5
 - Sklearn (http://scikit-learn.org/stable/)

## Run
 - `Algorithms.py` will extract features using `corpus.txt` from `dataset`
 - `Comparison.py` will use extracted features to classify documents

## Output
 - W: The word (Feature) 
 - Score of Algorithm: Score of W in the algorithm
 - Main Domain: Name of the domain in which W gets the highest score
 - Score of Main Domain: Score of w in the main domain
