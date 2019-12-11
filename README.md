# Supervised Learning Project

## Finding Donors for CharityML
This project was completed as part of the course requirements of Udacity's Machine Learning Introduction Nanodegree certification.

The complete code and analysis rationale can be found in [this browser friendly version](https://github.com/marcellovictorino/MLIND_Project1_SupervisedLearning/blob/master/finding_donors.html) (download and open with your favorite browser) or the actual [jupyter notebook](https://github.com/marcellovictorino/MLIND_Project1_SupervisedLearning/blob/master/finding_donors.ipynb).

### Setup

This project requires **Python >= 3.6** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)


### Overview

Analysis of the [modified version of the 1994 US census dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income), having approximately 45,000 data points and 14 features. Particularly, investigating if there are any factors that could be used to model individuals' yearly income.

The goal is to accurately identify an individuals' income, so the charity can focus their time and effort reaching out to people who are more likely to donate to their cause. Based on the fictitous case of CharityML, they have found from previous experience their target audience is people earning U$50k or more per year.

Since the data is labelled, having the feature `income` informing the individual's yearly income class: >50k or <= 50k, this is clearly a case to apply Supervised Learning for Classification.

After wrangling the data, applying transformation to normalize some features, and performing feature engineering, the final dataset goes from 14 to 103 features. The data is also normalized (scaled between 0 and 1) since it improves modeling performance and convergence.

The data is divided using an 80/20 split for training/testing. Six models were trained usig default parameters so we can perform model evaluation (accuracy, precision, recall, and Fbeta score) to identify the best candidate, before performing hyperparameter tuning in order to optimize it (using random grid search with 5-fold cross-validation on the training split only).

Finally, the best model is further investigated in order to evaluate its feature importance: allowing to better interpret the most impactful features identified by the model, and to double-check if the results are reasonable.
> The exercise of model interpretability is a critical step to better communicate results, helping to gain the trust from decision makers and build support and confidence around the model.

### Technologies Used
+ **Python:**
    - Pandas, Numpy, Matplotlib, Scikit-Learn, Jupyter Notebook
+ **Machine Learning:**
    - Logistic Regression, Random Forest, Extra Trees Classifier, AdaBoosting, Gradient Boosting, Gaussian Naive Bayes
    - Hyperparameter tuning using Random Grid Search
