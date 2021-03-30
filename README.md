# General Assembly Capstone Project: Kiva Loans

This repository contains the code for my Capstone project on [Kiva's open source dataset](https://www.kiva.org/build/data-snapshots), undertaken as part of General Assembly's Data Science Immersive course between November 2020 and February 2021. I chose this project to use my data science skills for the social good and wanted to work with humanitarian data. 

Kiva is an organization that allows people to lend small amounts of money via the Internet to help microfinance organizations in the developing world. These local microfinance organizations help local business people to post profiles and business plans on Kiva. Once online, lenders select profiles and business plans to fund. This is a crowdfunding website, meaning each loan has a campaign period in which borrowers have to convince lenders to fund their loan all the way up to the fundraising target. If the target isn't reached, the money is given back to the lenders. This project consisted in predicting loans which would 'expire', or in other words those who won't reach their fundraising target, in order to subsequently promote them on the front page of the website. These 'expired' loans consitute 4.5% of total loans, making this a severely imbalanced classification problem.

## Problem Statement

While doing some high-level EDA on the data, I had several considerations in mind:

-  Are lenders concerned by default or delinquency rates when deciding whether to fund a loan or not? 

Kiva has data about each borrowers microfinancial bank, more specifically:
 1. Default rates, or the percentage of ended loans which have failed to repay (measured in dollar volume, not units).
 2. Delinquency rates, or the amount of late payments divided by the total outstanding principal balance Kiva has with the bank. 
  
 These are two indicators describing a banks quality when paying back the loan to the lenders, which impact the lenders welfare.
 
-  If it isn't these indicators that lenders are concerned by when funding a loan, then what are they concerned by?
 
- If it isn't these indicators that lenders are concerned by when funding a loan, we could create a model to try and predict the loans that are expired to subsequently promote them ! Why?

If lenders do care about their own welfare when making these funding decisions, then there is not much point in promoting these loans as it would just be grouping high default rate loans which people wouldn't want to fund anyway due to their preferences.

If lenders don't care about their own welfare when making these decisions, then the problem is in Kiva's control and could be fixable. Supposing that loan amount is the main predictor for not having your loan funded, we could for example put an amount ceiling if your loan is predicted as 'expired' in order to have this loan promoted. 

In this latter case, the group would be uncorrelated with default rates. This means that there's some borrowers who didn't get their loan who would have been able to repay that loan. This model could help the people with characteristics likely in making their loan 'expired' but who could have repayed that loan and been happy with it.

### 2 Projects

To see if default rates impacted these decisions I decided to conduct a side project by scraping some data from Kiva's website and concatenating it to the main dataset. However, for this side project I resulted in having default rate data for less banks than the total amount of banks in the main dataset. This meant that this default rate analysis had to be done with a subset of the main dataset. I had to clean and model two different datasets. This is why there is 2 directories above, one for the main project and one for the side project. This side project follows most of the steps the main project has taken to clean, display and model the data.

### Prerequisites

To run the code, there are a number of dependencies. You may need to pip install the following:

```python
Pandas
MatplotLib
NumPy
DateTime
BeautifulSoup
SciKit Learn
```

Underneath are the four main parts of these projects with examples of what I did. 

## [Part 1: Data Cleaning](https://github.com/nicolas1998v/KivaCapstoneProject/blob/main/Main%20Project/Data%20Cleaning.ipynb)

In this part, I performed preliminary data munging and cleaning of the data.
  
### 1. Removing nulls. 

19 columns out of 34 columns contained null values inside them which needed to be removed for modeling. 

<img width="1021" alt="Screenshot 2021-03-27 at 22 27 30" src="https://user-images.githubusercontent.com/57761032/112736576-a1837c00-8f4b-11eb-93d7-6198db76ef55.png">
 
  ### 2. Reducing value counts. 

4 columns had many values in them. For example, the borrower genders column had nearly 25 thousand values. Since they are of type 'object', these values need to be dummified before modeling. A column with 25 thousand values will result in 25 thousand columns after dummification. This large amount of columns isn't optimal for modeling as it would incur computational difficulties that my computer wouldn't be able to overcome. In addition, it makes your columns values easier to understand. This is why I reduced the number of values for these 4 object columns.  
  
 <img width="1112" alt="Screenshot 2021-03-27 at 22 17 29" src="https://user-images.githubusercontent.com/57761032/112736408-3d13ed00-8f4a-11eb-8b1c-c583258a3afc.png">

For this specific example, I managed to reduce it to 6 values.

 <img width="1009" alt="Screenshot 2021-03-27 at 22 18 54" src="https://user-images.githubusercontent.com/57761032/112736430-6d5b8b80-8f4a-11eb-8b72-f687c3fdb557.png">

### 3. Feature Extractions.

I created 4 features out of this dataset. Here, I create a feature out of the borrower_genders column mentioned above that counts the number of males per loan.
 
 <img width="856" alt="Screenshot 2021-03-27 at 22 12 30" src="https://user-images.githubusercontent.com/57761032/112736289-8c0d5280-8f49-11eb-87ad-988f169c6070.png">

I created a feature named ***campaign duration***, taking the difference between the posted_time and planned_expiration_time columns.

<img width="945" alt="Screenshot 2021-03-27 at 22 31 55" src="https://user-images.githubusercontent.com/57761032/112736648-3f774680-8f4c-11eb-8aa6-7a67b119ad1b.png">

### 4. Removing unnecessary rows and columns.

Since Kiva doesn't provide documentation for its open-souce dataset, there were some columns which I had trouble understanding their meaning, and I had to do some investigation to realise they were not suitable for the analysis. Please look at the notebook if you want to see which columns I removed and my underlying thought process behind this decision.
Some loans were still in their crowdfunding campaign, so I removed these.

## [Part 2: EDA and Preprocessing]( https://github.com/nicolas1998v/KivaCapstoneProject/blob/main/EDA%20-%20NLP.ipynb)

Next, I visualised the data, looking at each values correlation with the target variable. 
 
This shows how borrowers prefer to fund females than males.
 
<img width="1010" alt="Screenshot 2021-03-27 at 22 37 47" src="https://user-images.githubusercontent.com/57761032/112736759-11decd00-8f4d-11eb-9cf8-8e8a4d19dd9e.png">

I then performed Natural Language Processing on the text variables with CountVectorizer, and dummified the object columns.

The CountVectorizer produced these words out of the description column.

<img width="955" alt="Screenshot 2021-03-27 at 22 38 27" src="https://user-images.githubusercontent.com/57761032/112736785-2fac3200-8f4d-11eb-852c-2aa62027c88a.png">
 
 ## [Part 3: Modeling](https://github.com/nicolas1998v/KivaCapstoneProject/blob/main/Modelling.ipynb) 
 
 This is a binary classification problem, predicting the 'Status' label. I set the 'Expired' value as positive and the 'Funded' value as Negative.

### Success Metric

The score we want to maximise is Precision. Since the use case scenario is promoting future 'expired' loans on the front page of the website, we want to make sure to predict only these loans, and minimise those that are predicted 'expired' too but would have been actually funded either way - or False Positives. Minimising False Positives means increasing Precision.

### Methods

For all of the following methods, I decided to test 2 algorithms : Logistic Regression and Random Forests.

I decided to first instantiate two basic models to have a baseline score I had to beat.

<img width="1020" alt="Screenshot 2021-03-27 at 23 03 35" src="https://user-images.githubusercontent.com/57761032/112737223-aac31780-8f50-11eb-8196-d1d932b53825.png">

Only 4.5% of labels were positive. I tried two methods to cope with the severe class imbalance.

1. Cost-Sensitive Method. I used Grid-Search to tune the 'class_weight' parameter in order to give more importance to the minority class.

<img width="1010" alt="Screenshot 2021-03-27 at 22 44 19" src="https://user-images.githubusercontent.com/57761032/112736882-faecaa80-8f4d-11eb-8904-68c9968778f7.png">

2. Sampling methods. Sampled the training set to subsequently train with better class label percentages.

<img width="1128" alt="Screenshot 2021-03-27 at 23 02 38" src="https://user-images.githubusercontent.com/57761032/112737203-8bc48580-8f50-11eb-9cd9-a828f6ad7d0c.png">

With 0.854 in test set precision, the basic model out-performed the other 2 methods(Cost-Sensitive best score: 0.842. Sampling best score: 0.740). Now, let's see how its confusion matrix looks like.

### Model Performance.

<img width="1024" alt="Screenshot 2021-03-27 at 22 45 35" src="https://user-images.githubusercontent.com/57761032/112736895-27082b80-8f4e-11eb-9c03-6822407c6830.png">

This chart shows the results of the best model. On the far left we have the y-axis with the label values, 0 for 'Funded' and 1 for 'Expired'. For both of these charts, we have:
- Loans that were originally funded and were predicted as funded on the top left of the chart, or True Negatives.
- Loans that were originally expired and were predicted as funded on the bottom left of the chart, or False Negatives.
- Loans that were originally funded and were predicted as expired on the top right of the chart, or False Positives.
- Loans that were originally expired and were predicted as expired on the bottom right of the chart, or True Positives.

The left hand chart shows the training set, which has near to perfect scores due to its inherent nature of being the training set. 
The right hand chart shows the testing set. The value we want to minimise is the False Positives on the top right of the chart. 

To calculate the proportion of False Positives among all the loans predicted as positive, we can do the following calculation:
84/(494+84) = 0.145. 

Hence, 14.5% of Positive loans are still False Postives. A next goal could be to bring that down to 10% and then 5% to increase precision further.

## [Part 4: Presentation]( https://docs.google.com/presentation/d/18hdJlMiIoCoKHjRcSIvIPgoN-mT_E5lz_FUGBUjFQaU/edit#slide=id.p)

- Hosted a short, well rehearsed presentation of your project for a non-technical audience. 
- Covered goals, success criteria, data, approach, basic description of model, findings, risks/limitations, impact, next steps and conclusions.

## Conclusion

### Side Project
Default rates aren't the reason why lenders don't fund loans. This is due to 3 insights:

1. The EDA didn't show a strict downward trend in funded percentages when increasing default rates. On the second half of this graph, as we keep increasing default rates, we still have high funded loans per default rates. There might be a small downward trend but this may not be enough to be statistically significant.

<img width="553" alt="Screenshot 2021-03-29 at 20 15 22" src="https://user-images.githubusercontent.com/57761032/112887808-82135d00-90cb-11eb-8fb9-a70afd31d396.png">

2. Model results weren't better. The best model came back with 0.844 in test set precision, down 1 point from the main model precision (0.854).

3. Inference graphs from these models didn't have default rates column as high predictors. 

### Main Project

The results from the main project models were 0.854 in test set precision, with 14.5% of False Positives. 
Loan Amounts, Lender Term, Genders and Sector are the main predictors. 
So this group I promote on the website's front page could have a cap in loan amounts and lender terms to make them more attractive to lenders.

Thank you for you attention.
