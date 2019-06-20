import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt4Agg')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# import data
data = pd.read_csv('/home/bhart/Documents/ML_projects/examples/churn_analysis/data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# a lot of data here is catagorical, let's use pd.get_dummies to fix that
dummies = pd.get_dummies(data[[
	'gender', 'SeniorCitizen', 'Partner', 'Dependents',
	'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
	'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
	'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
	'PaymentMethod', 'Churn'
]])

# add charges back into dataset and replace blanks with 0 and convert from string to numeric
data = dummies.join(data[['MonthlyCharges', 'TotalCharges']])
data['TotalCharges'] = data[['TotalCharges']].replace([' '], 0)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

# now let's take a look at the data
plt.scatter(
	data['tenure'],
	data['MonthlyCharges'],
	c=data['Churn_Yes']
)
plt.xlabel('Customer Tenure (Months)')
plt.ylabel('Monthly Charges')
plt.show()

# the plot doesn't provide any concrete insights. Let's use Logistic Regression to see if we can predict churn.
x_select = ['SeniorCitizen', 'tenure', 'gender_Female', 'gender_Male', 'Partner_No',
	'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
	'PhoneService_Yes', 'MultipleLines_No',
	'MultipleLines_No phone service', 'MultipleLines_Yes',
	'InternetService_DSL', 'InternetService_Fiber optic',
	'InternetService_No', 'OnlineSecurity_No',
	'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
	'OnlineBackup_No', 'OnlineBackup_No internet service',
	'OnlineBackup_Yes', 'DeviceProtection_No',
	'DeviceProtection_No internet service', 'DeviceProtection_Yes',
	'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
	'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
	'StreamingMovies_No', 'StreamingMovies_No internet service',
	'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
	'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
	'PaymentMethod_Bank transfer (automatic)',
	'PaymentMethod_Credit card (automatic)',
	'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
	'MonthlyCharges', 'TotalCharges'
]
X_train, X_test, Y_train, Y_test = train_test_split(data[x_select], data['Churn_Yes'])
clf = LogisticRegression(solver='lbfgs', max_iter=1000)
clf.fit(X_train, Y_train)

# lets look at the confusion matrix and accuracy
accuracy = accuracy_score(Y_test, clf.predict(X_test))
print('Accuracy: {}'.format(accuracy))
confusion = confusion_matrix(Y_test, clf.predict(X_test))
confusion_norm = confusion / confusion.astype(np.float).sum(axis=1)
print('confusion matrix: \n{}'.format(confusion_norm))

# so at this point we could try and improve the accuracy of the model, but lets stop and ask ourselves what our goal
# is here. We want to know if a customer will churn, but that doesn't provide a lot of insight. It would be nice to
# know what is causing them to churn and where to spend money to keep customers. Lets take a look at Survival Analysis.
# Logistic regression assigns a probability to each observation that describes how likely it is to belong is a class
# (churn or no churn). For the case above (binary classification) we assign a class based on which side of 0.5 the
# probability falls. In any large group of customers there is going to be segmentation where some of them will churn
# and others won't. What we'd like to know is the churn probability of each group and when that churn will occur.
# This is why we use Survival Analysis as we can use a set of methods that determine the probability of a customer
# churning over time (developed for use in life sciences and pharma research for patient death, Yay). There are a bunch
# of tools and statistical methods involved in Survival Analysis but lets just use the Cox Proportional Hazards Method.
# First lets import the lifelines library
from lifelines import CoxPHFitter
from lifelines.utils import median_survival_times, qth_survival_times
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from tqdm import tqdm
from tqdm import t

# Now we have to remove all the yes no pairs and just keep the yes's. We also remove gender_male,
# Contract_month_to_month, basically anything non binary.
x_select = ['SeniorCitizen', 'tenure', 'gender_Female',
	'Partner_Yes', 'Dependents_Yes',
	'PhoneService_Yes', 'MultipleLines_Yes',
	'InternetService_DSL', 'InternetService_Fiber optic',
	'OnlineSecurity_Yes',  'OnlineBackup_Yes',
	'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
	'StreamingMovies_Yes', 'Contract_One year',
	'Contract_Two year', 'PaperlessBilling_Yes',
	'PaymentMethod_Bank transfer (automatic)',
	'PaymentMethod_Credit card (automatic)',
	'PaymentMethod_Electronic check',
	'MonthlyCharges', 'TotalCharges', 'Churn_Yes'
]

# we can move straight on to splitting and fitting
cph = CoxPHFitter()
cph_train, cph_test = train_test_split(data[x_select], test_size=0.2)
cph.fit(cph_train, 'tenure', 'Churn_Yes')

# now lets look at a summary of cph
cph.print_summary()

# awesome!!!
# There are a few important things to notice about this output.
# 1. We can see the number of observations listed as n=5634 right at the top of the output, next to that we have our
#   number of events (churned customers).
# 2. We get the coefficients of our model. These are very important and they tell us how each feature increases risk,
#   so if it’s a positive number that attribute makes a customer more likely to churn, and if it is negative then
#   customers with that feature are less likely to churn.
# 3. We get significance codes for our features. A very nice addition!
# 4. We get the concordance. Our model has a concordance of .929 out of 1, so it’s a very good Cox model. We can use
#   this to compare between models, kind of like accuracy in Logistic Regression.

# lets actually plot all of this to get a better picture
cph.plot()
cph.plot_covariate_groups('TotalCharges', values=[0,4000], cmap='coolwarm')
# you can see in the survival curve plot that customers that have Total charges closer to 0 are at a higher risk of
# churning compared to those with charges closer to 4000.

# now lets do some churn prediction now that we have some useful insights into what makes customers churn.
# lets take all the non churners as we can't retain those who have already churned, these are called censored_subjects
# sticking to Survival Analysis lingo.
censored_subjects = data.loc[data['Churn_Yes'] == 0]

# now we can predict their unconditioned survival curves
unconditioned_sf = cph.predict_survival_function(censored_subjects)
# these are unconditioned because we will predict some churn before the customers current tenure time.

# lets condition the above prediction
conditioned_sf = unconditioned_sf.apply(lambda c: (c/c.loc[data.loc[c.name, 'tenure']]).clip_upper(1))

# now we can investigate customers to see how the conditioning has affected their survival over the baseline rate
subject = 12
unconditioned_sf[subject].plot(ls="--", color="#A60628", label="unconditioned")
conditioned_sf[subject].plot(color="#A60628", label="conditioned on $T>58$")
plt.legend()
# we can see that cust 12 is still a customer after 58 months, which means cust 12's survival curve drops slower than
# the baseline for similar custs without that condition.

# the predict_survival_function has created a metrix of survival probabilities for each remaining customer at each
# point in time. what we need to do now is use that to select a single value as  prdiction for how long a customer
# will last. Lets use the the median.
# predictions_50 = median_survival_times(conditioned_sf)
likelihood_cutoff = 0.5
predictions_50 = qth_survival_times(likelihood_cutoff, conditioned_sf) # same as above but specifying the %
# this gave us a single row with the month number where the customer has a 50% likelihood of churning.

# Lets join it to some data to investigate.
values = predictions_50.T.join(data[['MonthlyCharges', 'tenure']])
values['RemainingValue'] = values['MonthlyCharges'] * (values[likelihood_cutoff] - values['tenure'])
# now looking at the RemainingValue column we can see which customers would most affect our bottom line.

# Great, so we know which customers have the highest risk of churn and when they are likely to, but what can we do?
# lets take a look at our coefficients from earlier.
# we can see that the features that impact survival positively are 'Contract_One year', 'Contract_Two year',
# 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)'. Beyond these the results are
# insignificant. Lets compare customers with the features to understand the best place to spend money.
upgrades = ['Contract_One year',
            'Contract_Two year',
            'PaymentMethod_Bank transfer (automatic)',
            'PaymentMethod_Credit card (automatic)']
results_dict = {}
for customer in tqdm(values.index):
	actual = data.loc[[customer]]
	change = data.loc[[customer]]
	results_dict[customer] = [cph.predict_median(actual)]
	for upgrade in upgrades:
		change[upgrade] = 1 if list(change[upgrade]) == [0] else 0
		results_dict[customer].append(cph.predict_percentile(actual, p=likelihood_cutoff))
		change[upgrade] = 1 if list(change[upgrade]) == [0] else 0
results_df = pd.DataFrame(results_dict).T
results_df.columns = ['baseline'] + upgrades
actions = values.join(results_df).drop([likelihood_cutoff], axis=1)

# now we can calculate the difference between applying different features from the baseline
actions['CreditCard Diff'] = (
    actions['PaymentMethod_Credit card (automatic)'] -
    actions['baseline']
) * actions['MonthlyCharges']
actions['BankTransfer Diff'] = (
    actions['PaymentMethod_Bank transfer (automatic)'] -
    actions['baseline']
) * actions['MonthlyCharges']
actions['1yrContract Diff'] = (
    actions['Contract_One year'] - actions['baseline']
) * actions['MonthlyCharges']
actions['2yrContract Diff'] = (
    actions['Contract_Two year'] - actions['baseline']
) * actions['MonthlyCharges']

# lets take a look at the new actions data
pd.set_option('display.max_columns', 100)
print(actions.head(n=10))
actions.to_csv(index=False)

# So we now have a decent model (90+% concordance) were we can determine whether a churn intervention is worthwhile.
# lets now see how accurate it is.
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
probs = 1-np.array(cph.predict_survival_function(cph_test).loc[13])
actual = cph_test['Churn_Yes']
fraction_of_positives, mean_predicted_value = calibration_curve(actual, probs, n_bins=10, normalize=False)
ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" %("CoxPH",))
ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots (reliablility curve')

# great, we can see that our probabilities lie pretty close to the diagonal. We seem to be under predicting risk at the
# low end and over predicting risk at the top end. Let's get a numeric representation of he plot.
brier_score_loss(
    cph_test['Churn_Yes'], 1 -
    np.array(cph.predict_survival_function(cph_test).loc[13]), pos_label=1
)

# the lower the brier score loss the better but we need to get an idea of the loss at all possible tenure values, not
# just 13, so lets do that now
loss_dict = {}
for i in range(1, 73):
	score = brier_score_loss(
		cph_test['Churn_Yes'],
		1-np.array(cph.predict_survival_function(cph_test).loc[i]),
		pos_label=1
	)
	loss_dict[i] = [score]
loss_df = pd.DataFrame(loss_dict).T
fig, ax = plt.subplots()
ax.plot(loss_df.index, loss_df)
ax.set(xlabel='Prediction Time', ylabel='Calibration Loss', title='Cox PH Model Calibration Loss/Time')
ax.grid()
plt.show()

# we can see that our model is pretty well calibrated from about 5 to 25 months. A we forecast further into the future
# we start to get a lot more loss. The last step to make our model realistic is to take account of this calibration
# loss.
loss_df.columns = ['loss']
temp_df = actions.reset_index().set_index('PaymentMethod_Credit card (automatic)').join(loss_df)
temp_df = temp_df.set_index('index')
actions['CreditCard Lower'] = temp_df['CreditCard Diff'] - (temp_df['loss'] * temp_df['CreditCard Diff'])
actions['CreditCard Upper'] = temp_df['CreditCard Diff'] + (temp_df['loss'] * temp_df['CreditCard Diff'])
temp_df = actions.reset_index().set_index('PaymentMethod_Bank transfer (automatic)').join(loss_df)
temp_df = temp_df.set_index('index')
actions['BankTransfer Lower'] = temp_df['BankTransfer Diff'] - (.5 * temp_df['loss'] * temp_df['BankTransfer Diff'])
actions['BankTransfer Upper'] = temp_df['BankTransfer Diff'] + (.5 * temp_df['loss'] * temp_df['BankTransfer Diff'])
temp_df = actions.reset_index().set_index('Contract_One year').join(loss_df)
temp_df = temp_df.set_index('index')
actions['1yrContract Lower'] = temp_df['1yrContract Diff'] - (.5 * temp_df['loss'] * temp_df['1yrContract Diff'])
actions['1yrContract Upper'] = temp_df['1yrContract Diff'] + (.5 * temp_df['loss'] * temp_df['1yrContract Diff'])
temp_df = actions.reset_index().set_index('Contract_Two year').join(loss_df)
temp_df = temp_df.set_index('index')
actions['2yrContract Lower'] = temp_df['2yrContract Diff'] - (.5 * temp_df['loss'] * temp_df['2yrContract Diff'])
actions['2yrContract Upper'] = temp_df['2yrContract Diff'] + (.5 * temp_df['loss'] * temp_df['2yrContract Diff'])