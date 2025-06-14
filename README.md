# Restaurant_Classification

## Overview
This project examines a sample of a database of restaurants and various attribututes, including a collection of their reviews. The data is provided through a Kaggle Competition in which the objective is to improve upon a baseline model's accuracy of 0.77591 in classifying the type of restaurant. The data is primarily processed in Python using Pandas with simple machine learning models in scikit-learn used. 

Reference the below image for a list of restaurant types provided in the train and test set. 
<img src="/images/restaurant_classification.png" width="700">

## Notebook Index 
* [NB0_baseline.ipynb](NB0_baseline.ipynb) -  Initial word2vec / embedding model
* [NB1_optimization.ipynb](NB1_optimization.ipynb) - Improved model exploration

## Methods
The most effective tested approach was applying a TF-IDF Vectorizer to the reviews to feed into our models. Beyond that, the main features used were those that had the fewest null values. As most of these were booleans, one-hot encoding was used to set up the columns into usable features with `pd.get_dummies()`. The additional features underwent PCA for dimension reduction, selecting 20 principal components, as justified through the scree plot (Figure 1) and explained variance plot (Figure 2) which showed 25 components with eigenvalues >= 1 (to follow the [Kaiser Rule](https://en.wikipedia.org/wiki/Factor_analysis#Older_methods)) and around 22 components to explain 70% of the variance. 

## Initial Visualizations
Figure 1: Scree Plot of Pricipal Components
<img src="/images/scree_plot.png" alt="Figure 1" width="500">

Figure 2: Cumulative Variance Plot of Principal Components
<img src="/images/explained_variance_plot.png.png" alt="Figure 1" width="500">

## Modeling Approach and Results
With the constraint of resources and time (running on local setup), only 8 models were set up and tested, with results as follows: 

| #     | Model                                 | Features                                     | Accuracy     |
|-------|---------------------------------------|----------------------------------------------|--------------|
| 0     | Logistic Regression (baseline)        |word2vec, embeddings                          | 0.77591      |
| 1     | Logistic Regression                   |TF-IDF (reviews)                              | 0.79574      |
| 2     | Logistic Regression                   |TF-IDF (reviews), PCA                         | 0.80107      |
| 3     | Logistic Regression                   |TF-IDF (reviews), PCA, Numeric Attributes     | 0.80677      |
| 4     | AdaBoost (with Decision Tree CLF)     |TF-IDF (reviews), PCA, Numeric Attributes     | 0.67060      |
| 5     | LinearSVC                             |TF-IDF (reviews), PCA, Numeric Attributes     | 0.81514      |
| 6     | Random Forest                         |TF-IDF (reviews), PCA, Numeric Attributes     | 0.76835      |
| 7     | XGBoost                               |TF-IDF (reviews), PCA, Numeric Attributes     | 0.80030      |
| 8     | Ensemble*                             |TF-IDF (reviews), PCA, Numeric Attributes     | 0.82046      |

*Ensemble was a VotingClassifier with Logistic Regression, XGBoost, Random Forest, AdaBoost, and LinearSVC

The best model was Model 8, which was effectively a combination of all the previous models with all of the processed viable features.

## Further Remarks
Although the baseline model was unsuccessfully improved, there is additional potential in tweaking with careful parameter tuning of the TF-IDF used in conjunction. Additional hyper-parameter tuning could have been performed given more resources in order to create a more robust final ensemble model. There are still more attributes that could be explored in classification, to the benefit of the next steps of setting up basic deep learning models. 

The preliminary efforts were able to net a final accuracy score of 0.81768 on the test set that was held out in the Kaggle Competition, yielding a top 5 finish. 

![Final Finish](/images/restaurant_classification_rank.png)

## Sources
Data provided by UC San Diego's NLP Graduate course through Kaggle