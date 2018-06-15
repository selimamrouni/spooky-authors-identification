# Spooky-Authors-Identification

This repository presents our work during a project realized in the context of the IEOR 4523 Data Analytics Class at Columbia University.
This Natural Language Processing project comes originally from a [Kaggle competition](https://www.kaggle.com/c/spooky-author-identification). 

This is the description of the problem as detailed on Kaggle: "The competition dataset contains text from works of fiction written by spooky authors of the public domain: Edgar Allan Poe, HP Lovecraft and Mary Shelley. The data was prepared by chunking larger texts into sentences using CoreNLP's MaxEnt sentence tokenizer, so you may notice the odd non-sentence here and there. Your objective is to accurately identify the author of the sentences in the test set."

This is a multi-labels classification, with 3 labels:

- EAP: short text written by Edgar Allan Poe
- MWS: short text written by Mary Shelley
- HPL: short text written by HP Lovecraft

[Kaggle](https://www.kaggle.com) offers 2 datasets:
- One Training set (TR0): 19,579 extracts whose authors are known
- One Test set (TS0): 8,392 extracts whose author must be identified

Goal: For each extract, give probability to the potential authors (among the three mentioned above) to
determine which one is the most likely to be its author.

The evaluation metric is multilogloss. 

The project is decomposed in 3 parts:
- Feature Engineering: Extraction of features from the texts
- Pipelines Definition and Evaluation: Definition of the data pipeline and selection of 4 promising pipelines using K-Fold validation with 80% of TR0
- Final Model Evaluation: Selection of the best pipeline using the 80% of TR0 as training set and the remaining 20% of TR0 as testing set
- Submission: Training of the selected pipeline with the whole TR0 and submission of the prediction realized for TS0

The files are: 
- train.csv, test.csv: Data files. 
- spooky_project_cleaned.ipynb: This is the cleaned-up notebook with explanations. 
- spooky_project.ipynb: This is the notebook we used originally for the whole computing.
- Spooky Authors Identification - Presentation.pdf: This is the presentation we gave at the end of the project.
- screenShot: Folder with different screenshots of the project. 
- maskWorldCloud: Folder with the mask used to make the WorldClouds representations.
- NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt: File used for the sentiment analysis, it associates words with sentiment and the strength of the sentiment. 

## Feature Engineering

Originally coming from a [Kaggle](https://www.kaggle.com) contest, the dataset was clean without missing values. From the texts, we were able to generate two kind of features:
- Meta Features: related to the form of the text
- Text Features: related to the substance of the text

Then, this bunch of thousands (because of the use of [bag-of-words technic](https://en.wikipedia.org/wiki/Bag-of-words_model)) of numerical features has been scaled using min-max normalization. 

## Pipelines Definition and Evaluation

We used [sklearn pipeline class](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) in order to define the pipeline processing of the data. Indeed, because of the use of the text features, a huge bunch of features is generated (thousands...) and we needed first to reduce the number of features before applying the machine learning algorithm. 

We defined the pipelines composed of 3 elements:
- Number of kept features: (10, 20, 1/4 of the original features, 1/2 of the original features)
- Feature Selector: (Univariate Feature Selection ● Recursive Feature Elimination ● Principal Components Analysis)
- Predictive Model: (Logistic Regression ● KNN ● Random Forest ● Gradient Boosting ● ...)

Then, we took 80% of TR0 (so called tr1) and used 10-fold cross validation in order to compare the performances of the pipelines. 

## Final Model Evaluation

Among all the trained pipelines, the 10 best pipelines are selected for a final test. We used tr1 as training dataset and took the 20% remaining from TR0 (so called ts1) as test set. 

The best pipeline has been selected which was: (1/2 of the features (892), PCA, Logistic Regression) which achieves a log-loss of 0.61.

This pipeline was then trained with the whole TR0 (tr1 + ts1) and used to predict the author of the TS0 dataset.

## Author

* **Selim Amrouni** [selimamrouni](https://github.com/selimamrouni)
* **Leonardo Espenilla** [lespenilla](https://github.com/lespenilla)
* **Antoine Guincestre** [antoineguincestre](https://github.com/antoineguincestre)
* **Philippe Mizrahi** [pcamizrahi](https://github.com/pcamizrahi)




