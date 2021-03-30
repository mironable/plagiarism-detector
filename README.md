# Plagiarism detector project

This repository contains code and associated files for deploying a plagiarism detector using AWS SageMaker. This project was submitted in partial fulfillment of the
requirements for the Udacity Machine Learning Nanodegree.

## Project Overview

In this project a plagiarism detector is built that examines a text file and performs binary classification; labeling that file as either *plagiarised* or *not*, depending on how similar that text file is to a provided source text.

This project is broken down into three main notebooks:

**Notebook 1: Data Exploration**
* Load in the corpus of plagiarism text data.
* Exploration of the existing data features and the data distribution.

**Notebook 2: Feature Engineering**

* Clean and pre-process the text data.
* Define features for comparing the similarity of an answer text and a source text, and extract similarity features.
* Feature selection, by analyzing the correlations between different features.
* Create train/test `.csv` files that hold the relevant features and class labels for train/test data points.

**Notebook 3: Train and deploy a neural network in SageMaker**

* Upload train/test feature data to S3.
* Define a binary classification PyTorch model and a training script.
* Train the PyTorch model and deploy it using SageMaker.
* Evaluate the deployed classifier.

