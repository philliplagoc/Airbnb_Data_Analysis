# Analyzing What Features Affect Airbnb Listing Prices 

Airbnb is one of the largest marketplaces for providing lodging arrangements and tourism experiences around the world. There are many different data science projects one can do when it comes to looking at data from this company. In my case, this project arose from a take-home data challenge, and after talking with the recruiters, I can make my analysis public.


## My Goals
My primary goal in this data challenge was to **predict the price of a given listing**. Additionally, I wanted to look more into the different features in my data and see how they influenced price. 

In all, this was a **supervised learning regression problem**, where my primary goal was to predict the price of listings.


## Overview
For a more in-depth explanation of my methods, refer to the notebook. I will briefly go over the steps I did and why I did them.

### Preprocessing 
For preprocessing, I first dealt with missing data, which included dropping some features and encoding the missing data as its own category.

Afterwards, I transformed the categorical variables using ordinal encoding to retain some inherent ordering not immediately noticeable to us e.g. A+ ratings come before B ratings. The quantitative variables and the price column were log-transformed due to not being normal.

### Modelling
For modelling, I started used Linear Regression as a baseline, then proceeded to use XGBoost, GradientBoost, LightGBM, and AdaBoost. I bagged these models together by averaging their predictions.

Then, I moved on to reducing the feature space using PCA to combat the variability in the data i.e. there were lots of outliers and my models were overfitting. I repeated the above modelling process on the reduced feature space.

I also looked at using XGBoost in feature selection to determine which features were the most important, and a 5 layer neural network using L1 regularizers and the ReLU activation function just to see how a neural network might perform.

Finally, I encoded some features differently i.e. transformed quantitative variable to categorical and ordinally encoded it, and re-ran the modelling procedure on this newly encoded data. 

## Getting the Data and Using the Notebook
The data can be found [here](http://insideairbnb.com/get-the-data.html).
More specifically, I looked at Amsterdam listings that were recorded as of August 8th, 2019.

Packages include:
- `pandas` for data manipulation and cleaning
- `seaborn` and `matplotlib` for data visualization
- `numpy` for quick scientific computing
- `shapely`, `descartes` and `geopandas` for map visualizations
- `scipy` for statistical tests
- `datetime` and `statsmodels` for analyzing seasonality
- `sklearn`, `lightgbm` and `xgboost` for modelling

These packages can be installed via `pip install {package_name}`.

Running the notebook requires an IDE that can recognize and run `.ipynb` files, such as Jupyter Notebook.

## Conclusions 

| __Model__ 	| __Test MAE__ 	|
|:---------------------------------------------------------------	|--------------:	|
| Linear Regression w/out PCs 	| 0.2688 	|
| XGBoost w/out PCs 	| 0.2501 	|
| LGBM w/out PCS 	| 0.2527 	|
| GradientBoosting w/out PCs 	| 0.2517 	|
| AdaBoost w/out PCs 	| 0.2991 	|
| Averaging XGBoost, LGBM, GradientBoosting, AdaBoost w/out PCs 	| 0.2515 	|
| Linear Regression w/ PCs 	| 0.2764 	|
| XGBoost w/ PCs 	| 0.2738 	|
| XGBoost w/ Feature Selection (Best Threshold) 	| 0.2544 	|
| 4 Layer Neural Network w/ L1 Regularization 	| 0.2718 	|
| XGBoost w/out PCs on Newly Encoded Data 	| 0.2429 	|
| LGBM w/out PCS on Newly Encoded Data 	| 0.2483 	|
| GradientBoosting w/out PCs on Newly Encoded Data 	| 0.2472 	|
| AdaBoost w/out PCs on Newly Encoded Data 	| 0.2987 	|
| **Averaging XGBoost, LGBM, GradientBoosting, AdaBoost w/out PCs on Newly Encoded Data** 	| **0.2467** 	|

Given the slight increase in MAE when using my bagging technique on the newly encoded data, I will conclude that my best model for predicting Airbnb sales is bagging AdaBoost, GradientBoost, LGBM, and XGBoost together using the newly encoded data. However, one can argue that the other, simpler models perform just as well. I opt for the bagging technique because it still performs explicitly better and, if more features are added to the data, it would be able to capture those complexities. 

### Some Takeaways

| __Question__ 	| __Answer__ 	|
|:--------------------------------------------------------------------------------------------------------------------------------------------------------	|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| What's the relationship between ratings and price? 	| In terms of the median price, it would seem that regardless of the rating, the prices are the same. This could be due to ratings coming after a listing being priced. This could also be due to the fact that hosts want to make as much money as possible 	|

## Recommendations
For Airbnb, if you wanted to ensure that you'd get the most hosts to come to your website and to optimize guest booking these listings, there are several things I'd consider: 

One reason why my models performed as they did may be because **there are other features that are important but aren't in this dataset, such as a photo of the listing or a list of nearby events at the time of the listing**. Features like these could potentially influence price. If you wanted to help potential hosts make better decisions in pricing their homes, and thus get more guests, it would be worthwhile to consider such features. 

Another thing to consider would be using clustering techniques. Instead of using models like linear regression, one can also use a clustering algorithm, such as K-Means Clustering, to determine the different clusters of listings in an area, and help hosts determine which one their listing belongs to.


## Extensions and Further Explorations
- How has COVID-19 affected Airbnb?
- How can I make a model even more suitable to hosts?
- Can these trends be seen in other areas other than Amsterdam?
- What are the other ways to reduce variability in the dataset?

I also built a model that can help potential hosts predict the price of their listing. To create this model, I removed the features unknown to the host prior to putting their listing up e.g. ratings and built a model based on my analysis here. 

The Airbnb Housing Price Predictor model can be found in [this Github repo](https://github.com/philliplagoc/Airbnb-Housing-Price-Predictor).
