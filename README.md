# FeatureExtraction
Implementation of Information Gain, Mutual Information and Chi-Square Algorithms.

## Introduction
At first, we should extract documents categories. Next, we need some statistical analysis such as the number of documents containing a specific word. Then we can calculate the score of a word more easily. Finally, we can sort all scores and choose our features.

Algorithms I implemented include:
 * Information Gain
 * Mutual Information
 * Chi-Square

After Extracting Features we can use them to classify our documents. I use the normalized term frequency vector as a representation. In training step, I used SVM classifier with extracted vectors.

## Data
A part of Hamshahri Corpus was used as dataset. I extracted 100 most important features.

## Uses
 - [Numpy] (http://www.numpy.org/) version 1.14.5
 - [Sklearn] (http://scikit-learn.org/stable/)

## Run
 - `Algorithms.py` will extract features using `corpus.txt` from `dataset`
 - `Comparison.py` will use extracted features to classify documents

## Output
 - W: The word (Feature) 
 - Score of Algorithm: Score of W in the algorithm
 - Main Domain: Name of the domain in which W gets the highest score
 - Score of Main Domain: Score of w in the main domain
