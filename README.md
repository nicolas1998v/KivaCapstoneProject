# General Assembly Capstone Project: Kiva Loans

This repository contains the code for my Capstone project on the 2 million dataset [made available](https://www.kiva.org/build/data-snapshots) by Kiva, undertaken as part of General Assembly's Data Science Immersive course between November 2020 and February 2021. I chose this project as I would like to use my data science skills for the social good and wanted to find a project with humanitarian data. 

This project consisted in predicting loans which would 'expire', or in other words those who won't reach their fundraising target in the website, in order to subsequently promote them on the front page of the website. These 'expired' loans consitute 4.5% of total loans, making this a severely imbalanced classification problem.

## PROBLEM STATEMENT

While doing some high-level EDA on the data, I was asking myself 2 questions:
-  Are lenders concerned by default or delinquency rates when deciding whether to fund a loan or not? And if it's not this they're concerned by, then what is it ?
- If lenders aren't concerned by default rates, could we not create a model to try and predict the loans that are expired to subsequently promote them? 

If lenders do care about their own welfare when making these decisions, then there is not much point in promoting these loans as it would just be grouping high default rate loans which people wouldn't want to fund anyway due to their preferences.

If lenders don't care about their own welfare when making these decisions, then the problem isn't exogenous but endogenous, which is fixable. Supposing that loan amount is the main predictor for not having your loan funded, we could for example put an amount ceiling if your loan is predicted as 'expired' in order to have this loan promoted. Or if the predictors are more varied, then lenders could each find their own pick matching their preferences when checking the loans in this 'low quality' group.

In the latter case, the group would be uncorrelated with default rates. This means that there's some borrowers who didn't get their loan who would have been able to repay that loan. This model could help the people with characteristics likely in making their loan 'expired' but who could have repayed that loan and been happy with it.

### DEFAULT RATE SIDE PROJECT

To see if default rates impacted these decisions, I scraped some data from Kiva's website and concatenated it to the main dataset. However, in the process I resulted having default rate data for less banks than the amount of banks there was in the main dataset, meaning this default rate analysis had to be in a separate dataset, a subset of the main dataset. I had to clean and model two different datasets. This is why there is 2 folders above, one for the main project and one for the side project. This is a side project that gives greater insight to the main model and follows most of the steps the main project has taken to clean, display and model the data.

## SUCCESS METRIC

The Score we want to maximise is Precision. Since the use case scenario is promoting future 'expired' loans on the front page of the website, we want to make sure to predict only these loans, and minimise those that are predicted 'expired' too but would have been actually funded either way - or False Positives. Minimising False Positives means increasing precision.

Underneath are the four main parts of this experience with examples of what I did. 

## [Part 1: Cleaning](http://localhost:8888/notebooks/project/project-capstone/Capstone%20-%20Data%20Cleaning.ipynb)

- Perform preliminary data munging and cleaning of your data.
  
 1. Removing nulls. 

This is an example of when I removed nulls. Here, I realised that some of the nulls in the description_translated column where there because they were already in English in the decription column. Hence, I created a subset and then used loc with the index of the subset to remplace the nulls by the values in the description column.

<img width="1021" alt="Screenshot 2021-03-27 at 22 27 30" src="https://user-images.githubusercontent.com/57761032/112736576-a1837c00-8f4b-11eb-93d7-6198db76ef55.png">
 
  2. Reducing value counts. 

Some columns had way to high amounts of values in them. For example, the borrower genders column had nearly 25 thousand values.
  
 <img width="1112" alt="Screenshot 2021-03-27 at 22 17 29" src="https://user-images.githubusercontent.com/57761032/112736408-3d13ed00-8f4a-11eb-8b1c-c583258a3afc.png">

I managed to reduce it to 6 values.

 <img width="1009" alt="Screenshot 2021-03-27 at 22 18 54" src="https://user-images.githubusercontent.com/57761032/112736430-6d5b8b80-8f4a-11eb-8b72-f687c3fdb557.png">

3. Feature Extractions

Here, I create a feature out of the borrower_genders column mentioned above that takes the number of males per loan, thanks to this function I made.
 <img width="856" alt="Screenshot 2021-03-27 at 22 12 30" src="https://user-images.githubusercontent.com/57761032/112736289-8c0d5280-8f49-11eb-87ad-988f169c6070.png">

I create a feature named Campaign Duration, taking the difference between the posted_time and planned_expiration_time columns, and take just the days out of it.

<img width="945" alt="Screenshot 2021-03-27 at 22 31 55" src="https://user-images.githubusercontent.com/57761032/112736648-3f774680-8f4c-11eb-8aa6-7a67b119ad1b.png">

- Removing unnecessary rows and columns.

Since Kiva doesn't provide documentation, there were some columns which I had trouble understanding their meaning, and I had to do some investigation to realise they were not suitable for the analysis. Please look at the notebook if you want to see which columns I removed and my underlying thought process behind this decision.

## [Part 2: EDA and Preprocessing]( https://github.com/nicolas1998v/KivaCapstoneProject/blob/main/EDA%20-%20NLP.ipynb)
- Quantitatively visualise your data, looking at each values correlation with the target variable. 
 
   Produced this type of bar graph for every variable. This shows how borrowers prefer to fund females than males.
 
<img width="1010" alt="Screenshot 2021-03-27 at 22 37 47" src="https://user-images.githubusercontent.com/57761032/112736759-11decd00-8f4d-11eb-9cf8-8e8a4d19dd9e.png">

- Perform Natural Language Processing on the text variables, and dummify the object columns.

These are the words I had when using NLP on the description column.

<img width="955" alt="Screenshot 2021-03-27 at 22 38 27" src="https://user-images.githubusercontent.com/57761032/112736785-2fac3200-8f4d-11eb-852c-2aa62027c88a.png">
 
 ## [Part 3: Modelling](https://github.com/nicolas1998v/KivaCapstoneProject/blob/main/Modelling.ipynb) 

- Detail your model. 

Due to the severe class imbalance, I tried 2 methods to try and cope with this challenge: 

1. Cost-Sensitive Methods. Used Grid-Search to tune the 'class_weight' parameter in order to give more importance to the minority class.

<img width="1010" alt="Screenshot 2021-03-27 at 22 44 19" src="https://user-images.githubusercontent.com/57761032/112736882-faecaa80-8f4d-11eb-8904-68c9968778f7.png">

2. Sampling methods. Sample the training set to subsequently train with better class label percentages.

<img width="1128" alt="Screenshot 2021-03-27 at 23 02 38" src="https://user-images.githubusercontent.com/57761032/112737203-8bc48580-8f50-11eb-9cd9-a828f6ad7d0c.png">

3. However, the best model results came with the basic model. 

<img width="1020" alt="Screenshot 2021-03-27 at 23 03 35" src="https://user-images.githubusercontent.com/57761032/112737223-aac31780-8f50-11eb-8196-d1d932b53825.png">


- Evaluate model performance and discuss results. 

<img width="1024" alt="Screenshot 2021-03-27 at 22 45 35" src="https://user-images.githubusercontent.com/57761032/112736895-27082b80-8f4e-11eb-9c03-6822407c6830.png">

84/(494+84) = 0.145. 

Hence, 14.5% of loans are still False Postives, or loans that are of the 'funded' class but have been predicted as 'expired'. A next goal could be to bring that down to 10% and then 5%.

## [Part 4: Presentation]( https://docs.google.com/presentation/d/18hdJlMiIoCoKHjRcSIvIPgoN-mT_E5lz_FUGBUjFQaU/edit#slide=id.p)

- Host a short, well rehearsed presentation of your project for a non-technical audience. 
- Cover goals, success criteria, data, approach, basic description of model, findings, risks/limitations, impact, next steps and conclusions.

# CONCLUSION

In conclusion, the side project proved that default rates are not the reason why lenders don't fund loans. This is because :
1. The EDA didn't show a downward trend in funded percentages when increasing default rates.
2. Model results weren't better.
3. Inference graphs from these models didn't have default rates column as high predictors. 

So the model for the main project makes since in doing. 
The results from the main project models were 0.854 in test set precision, with 14.5% of False Positives. 
Loan Amounts, Lender Term, Genders and Sector seem to be the main predictors. So for example, this group I promote on the website's front page could have a cap in loan amounts and lender terms to make them more attractive to lenders.
