# Analyzing What Features Affect Airbnb Listing Prices 

Airbnb is one of the largest marketplaces for providing lodging arrangements and tourism experiences around the world. There are many different data science projects one can do when it comes to looking at data from this company. In my case, this project arose from a take-home data challenge, and after talking with the recruiters, I can make my analysis public.


## My Goals
My primary goal in this data challenge was to **predict the price of a given listing**. Additionally, I wanted to look more into the different features in my data and see how they influenced price. 

In all, this was a **supervised learning regression problem**, where my primary goal was to predict the price of listings.

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

### Some Takeaways

| __Question__ 	| __Answer__ 	|
|:--------------------------------------------------------------------------------------------------------------------------------------------------------	|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| What's the relationship between ratings and price? 	| In terms of the median price, it would seem that regardless of the rating, the prices are the same. This could be due to ratings coming after a listing being priced. This could also be due to the fact that hosts want to make as much money as possible 	|
| How many listings in each neighborhood? What is the most expensive neighborhood? | Most listings are in central Amsterdam. There are less listings towards the outskirts of Amsterdam. However, prices are relatively the same regardless of proximity towards the center of Amsterdam. Listings on the outskirts were priced as much, if not more than, listings in central Amsterdam. The prices here may reflect the idea that hosts want to capitalize on the idea that these housing projects are newer than those in central Amsterdam, and that there is less people in these areas since it is so new. One could also interpret this as the fact that hosts want to make as much money as they can from their listings, regardless of location. |
| Which model worked the best? | Averaging the predictions from XGBoost, LightGBM, GradientBoost, and AdaBoost got an MAE of 0.24. |

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
