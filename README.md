# General Assembly Capstone Project: Kiva Loans

This repository contains the code for my Capstone project on [Kiva's open microlending dataset](https://www.kiva.org/build/data-snapshots), undertaken as part of General Assembly's Data Science Immersive course between November 2020 and February 2021. I chose this project to use my data science skills for the social good and to work with humanitarian data. 

Kiva is an organization that allows people to lend small amounts of money to microfinance banks in the developing world. These organizations help local business people post their business plans on Kiva. Lenders can then select business plans to fund. This is a crowdfunding website, meaning each loan has a campaign period during which borrowers try to convince lenders to fund 100% of their loan. If the fundraising target isn't reached, the money is given back to the lenders. 

This project consists in predicting which loans might 'expire', or not reach their fundraising target. We want to identify them in order to promote them on the front page of the website. These 'expired' loans consitute 4.5% of all the loans, making this a severely imbalanced classification problem.

### Prerequisites
You'll need python 3 to run the notebooks. Then install the project's dependencies by typing: 

```bash
pip install -r requirements.txt
```

Underneath are the four main steps of each project, with examples of what I did. 

## [Part 1: Data Cleaning](https://github.com/nicolas1998v/KivaCapstoneProject/blob/main/Main%20Project/Data%20Cleaning.ipynb)

The dataset had 34 columns. In the end, I used 23 in the model. Many had to be cleaned. In addition, 5 of the columns were used to extract additional features. In the end, the model was trained on 939 features.
  
### 1. Removing nulls 

19 out of 34 columns contained null values inside them which needed to be removed for modeling. 

<img width="1021" alt="Screenshot 2021-03-27 at 22 27 30" src="https://user-images.githubusercontent.com/57761032/112736576-a1837c00-8f4b-11eb-93d7-6198db76ef55.png">
 
 ### 2. Reducing value counts 

4 out of 34 columns had too many values in them. For example, the **borrower_genders** column had nearly 25 thousand distinct values. Since they are string columns, these values need to be dummified before modeling. A column with 25 thousand values will result in 25 thousand columns after dummification. This large amount of columns makes learning about gender effects impossible. In addition, reducing value counts makes your columns values easier to understand. This is why I reduced the number of values for these string columns.  
  
 <img width="1112" alt="Screenshot 2021-03-27 at 22 17 29" src="https://user-images.githubusercontent.com/57761032/112736408-3d13ed00-8f4a-11eb-8b1c-c583258a3afc.png">

For this specific example, I managed to reduce it to 6 values.

 <img width="1009" alt="Screenshot 2021-03-27 at 22 18 54" src="https://user-images.githubusercontent.com/57761032/112736430-6d5b8b80-8f4a-11eb-8b72-f687c3fdb557.png">

### 3. Feature Extractions

In additon to cleaning up certain columns in the dataset, I extracted numerous new features from the **posted_time**, **planned_expiration_time**, **borrower_genders**, **loan_use**, **descriptions** and **tags** columns. 

For instance, I created a feature out of the **borrower_genders** column mentioned above that counts the number of males per loan.
 
 <img width="856" alt="Screenshot 2021-03-27 at 22 12 30" src="https://user-images.githubusercontent.com/57761032/112736289-8c0d5280-8f49-11eb-87ad-988f169c6070.png">

I also created a feature named **campaign duration**, taking the difference between the **posted_time** and **planned_expiration_time** columns.

<img width="945" alt="Screenshot 2021-03-27 at 22 31 55" src="https://user-images.githubusercontent.com/57761032/112736648-3f774680-8f4c-11eb-8aa6-7a67b119ad1b.png">

 **Description**, **loan_use** and **tags** were free text columns, so I counted the instances of each token - an NLP technique - using sklearn's CountVectorizer. 

### 4. Removing unnecessary rows and columns

Since Kiva doesn't provide documentation for its open dataset, there were some columns which I had trouble understanding. After some investigation, I realised they were not suitable for the analysis. Please look at the notebook if you wish to see which columns I removed, and my underlying thought process behind each decision.

Some loans were still in their crowdfunding campaign, so I removed these rows as well.

## [Part 2: EDA and Preprocessing]( https://github.com/nicolas1998v/KivaCapstoneProject/blob/main/EDA%20-%20NLP.ipynb)

Next, I visualised the data, looking at each feature's correlation with the target variable. 
 
I discovered that borrowers prefer to fund females than males.
 
<img width="1010" alt="Screenshot 2021-03-27 at 22 37 47" src="https://user-images.githubusercontent.com/57761032/112736759-11decd00-8f4d-11eb-9cf8-8e8a4d19dd9e.png">

The CountVectorizer produced these words out of the description column.

<img width="955" alt="Screenshot 2021-03-27 at 22 38 27" src="https://user-images.githubusercontent.com/57761032/112736785-2fac3200-8f4d-11eb-852c-2aa62027c88a.png">
 
 ## [Part 3: Modeling](https://github.com/nicolas1998v/KivaCapstoneProject/blob/main/Modelling.ipynb) 
 
 This is a binary classification problem, predicting the 'Status' label. I set the 'Expired' value as positive and the 'Funded' value as Negative.

### Success Metric

The score we want to maximise is Precision. Since the use case is promoting loans at risk of expiring on Kiva's front page, we want to avoid promoting False Positives: loans that we incorrectly predicted as being at risk of expiring, but which would actually have been funded. To increase Precision, we need to minimise False Positives.

### Model Training and Evaluation

To solve this binary classification problem, I used Logistic Regression and Random Forests.

First, I decided to train these models without tackling the class imbalance problem, in order to have a baseline score to beat. 

<img width="996" alt="Screenshot 2021-04-08 at 18 39 39" src="https://user-images.githubusercontent.com/57761032/114072116-cc4dc880-9899-11eb-976a-bcbf00beb1f9.png">

 Next, I needed to tackle the severe class imbalance. I tried two methods to cope with the fact that 4.5 % of the labels were positive. I researched these methods in the book "Imbalanced Classification with Python: Better Metrics, Balance Skewed Classes, Cost-Sensitive Learning" by Brownlee.

1. Cost-Sensitive Method. I used Grid-Search to tune the 'class_weight' parameter in order to give more importance to the minority class.

<img width="1010" alt="Screenshot 2021-04-08 at 18 51 24" src="https://user-images.githubusercontent.com/57761032/114073653-6eba7b80-989b-11eb-8bc3-64deafa8fc0f.png">

2. Sampling methods. Sampled the training set to subsequently train with better class label percentages.

<img width="1011" alt="Screenshot 2021-04-08 at 18 52 21" src="https://user-images.githubusercontent.com/57761032/114073764-8eea3a80-989b-11eb-9b3d-8183286ba85c.png">

With 0.879 in test set precision, the basic model out-performed the other 2 methods(Cost-Sensitive best score: 0.869. Sampling best score: 0.734). Now, lets look at False Positives.

**Confusion Matrix**

<img width="591" alt="Screenshot 2021-04-08 at 18 40 19" src="https://user-images.githubusercontent.com/57761032/114072192-e12a5c00-9899-11eb-8823-769d41fe5302.png">

This chart shows the results of the best model. On the far left we have the y-axis with the label values, 0 for 'Funded' and 1 for 'Expired'. For both of these charts, we have:
- Loans that were originally funded and were predicted as funded on the top left of the chart, or True Negatives.
- Loans that were originally expired and were predicted as funded on the bottom left of the chart, or False Negatives.
- Loans that were originally funded and were predicted as expired on the top right of the chart, or False Positives.
- Loans that were originally expired and were predicted as expired on the bottom right of the chart, or True Positives.

Calculating the proportion of False Positives per the total amount of predicted postives, we have:
- 77 / (77 + 560) = 0.1209

Hence, only 12% of the loans our model thinks will expire are False Postives. That's pretty good! The next goal could be to bring that down to 10% and then 5% to increase precision further.

## [Part 4: Presentation]( https://docs.google.com/presentation/d/18hdJlMiIoCoKHjRcSIvIPgoN-mT_E5lz_FUGBUjFQaU/edit#slide=id.p)

I have put together a short presentation of the project that covers goals, success criteria, data, approach, basic description of model, findings, risks/limitations, impact, next steps and conclusions.

## Side Project: Default Rates

While doing EDA on the data, I had one key question in mind:

-  Are lenders concerned by a bank's default or delinquency rates when deciding whether to fund one of their borrower's loans? 

To define those rates:
 1. Default rates, or the percentage of ended loans which have failed to repay (measured in dollar volume, not units).
 2. Delinquency rates, or the amount of late payments divided by the total outstanding principal balance Kiva has with the bank. 
  
 These two numbers tell a borrower if they're going to get payed back or not, which could impact their decision to lend.

The problem is that Kiva does not provide default and delinquency rates in the dataset, so I had to scrape them from their website. The scraping code is [here](https://github.com/nicolas1998v/KivaCapstoneProject/blob/main/Side%20Project/Data%20Appropriation.ipynb).

If lenders do care about the banks default rate, then there is not much point in promoting loans from these banks.

If they don't, then the problem is in Kiva's control and could be fixable. To check this, we need to measure the correlation between default rates and funding rates. 

### Main Project vs Side Project

To analyse the impact of default rates, I scraped some data from Kiva's website and concatenated it to the main dataset. However, I was only able to get default rates for 47% of the banks in the main dataset. This meant that the default rate analysis had to be done on a subset of the data. This is why there are 2 directories above, one for the main project and one for the side project. The side project follows the same steps as the main project has taken to clean, display and model the smaller dataset.

## Conclusion

### Main Project

The results from the main project models were 0.879 in test set precision, with 12% of False Positives. In other words, if our model was deployed to select promotions for Kiva's front page, it would make good selections 88% of the time. 

**Loan Amounts**, **Lender Term**, **Genders** and **Sector** were the main predictors. 

### Side Project

Additionally, I found that bank default rates were not a major driver of expiration probability.

1. The EDA didn't show an inverse relationship between default rates and funding probability. As you can see in the graph, banks with high default rates still had high funding rates.

<img width="553" alt="Screenshot 2021-03-29 at 20 15 22" src="https://user-images.githubusercontent.com/57761032/112887808-82135d00-90cb-11eb-8fb9-a70afd31d396.png">

2. Model results weren't better when we added this extra feature. The best model that included the feature came back with 0.852 in test set precision, down 2.5 points from the best model precision without it (0.879).

3. Inference graphs from these models didn't have default rates column as high predictors. 


Thank you for you attention.
