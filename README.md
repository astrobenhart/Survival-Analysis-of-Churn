# Survival-Analysis-of-Churn
Using techniques developed in the area of Survival Analysis (usually used for drug trials) to predict Telco customer churn.

This is actually a pretty useful idea as customers churning is anologous to patients dying.

### requirements
- pandas
- numpy
- tqdm (optional)
- matplotlib (optional)
- lifelines
- sklearn

### Notes
- This code is my attempt to get a foot hold in this area. As such it is basically a copy of Carl Dawson's [Churn Prediction and Prevention](https://towardsdatascience.com/churn-prediction-and-prevention-in-python-2d454e5fd9a5) artical. All created for the creation of this code should go to him and his sources.
- It should also be noted that in the real world data is messy, so the data used here, avaliable via [Kaggle's Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) website, is only an idealised example. From expereince, data, especially Telco or large company data is messy and a large portion of the development for any project like this is involved in getting good data, especially if it needs to be labelled.
