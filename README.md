# UDACITY STARBUCK'S CAPSTONE CHALLENGE

## Dataset Overview
- The dataset is generated using a program that simulates customer purchasing behavior and the impact of promotional offers on those decisions.
- Each simulated individual possesses hidden traits influencing their buying patterns, coupled with observable characteristics. These individuals engage in various events, encompassing offer reception, offer viewing, and transaction completion.
- Notably, the dataset doesn't focus on tracking specific products; rather, it records the transaction or offer amounts.
- Offers come in three types: Buy-One-Get-One (BOGO), Discount, and Informational. BOGO necessitates a specified spending threshold to receive a reward equal to that threshold. In a Discount offer, users receive a reward corresponding to a fraction of their expenditure. Informational offers lack rewards and required spending thresholds. Offers can be conveyed via multiple communication channels.
- The core objective is to utilize this data to discern which user segments respond most effectively to each offer type and optimize how these offers are presented.

## Data Dictionary
**profile.json**
Rewards program users (17,000 users x 5 fields)

- gender: (categorical) M, F, O, or null
- age: (numeric) Missing values are encoded as 118
- id: (string/hash)
- became_member_on: (date) Format: YYYYMMDD
- income: (numeric)

**portfolio.json**
Offers sent during a 30-day test period (10 offers x 6 fields)

- reward: (numeric) Money awarded for the amount spent
- channels: (list) web, email, mobile, social
- difficulty: (numeric) Money required to be spent to receive the reward
- duration: (numeric) Time the offer is open, in days
- offer_type: (string) bogo, discount, informational
- id: (string/hash)

**transcript.json**
Event log (306,648 events x 4 fields)

- person: (string/hash)
- event: (string) offer received, offer viewed, transaction, offer completed
- value: (dictionary) Different values depending on the event type
- offer id: (string/hash) Not associated with any "transaction"
- amount: (numeric) Money spent in a "transaction"
- reward: (numeric) Money gained from "offer completed"
- time: (numeric) Hours after the start of the test

Certainly, you can list the libraries used in a README format as follows:

**Libraries Used:**

1. Pandas (imported as `pd`)
2. Numpy (imported as `np`)
3. Math
4. JSON
5. Matplotlib (imported as `plt`)
6. Seaborn (imported as `sns`)
7. Scikit-Learn Linear Regression (from `sklearn.linear_model`)
8. Scikit-Learn Train-Test Split (from `sklearn.model_selection`)
9. Scikit-Learn Metrics - R-squared Score and Mean Squared Error (from `sklearn.metrics`)
10. Scikit-Learn Preprocessing - MinMaxScaler and StandardScaler (from `sklearn.preprocessing`)
11. Warnings

These libraries are used for various tasks, including data manipulation, visualization, machine learning modeling, and data preprocessing.

Blog : https://medium.com/@trandangtin91/starbucks-capstone-challenge-50e0b61682a8

**File Description**
- Udacity_DS_Pj3/Capstone Project.ipynb : Source code for data discovery and ML model processing
- Udacity_DS_Pj3/README.md : Summary for mainpoint technical
- Udacity_DS_Pj3/data/portfolio.json 
- Udacity_DS_Pj3/data/profile.json 
- Udacity_DS_Pj3/data/transcript.json

**Licensing, Authors, Acknowledgements, etc.**
Data for coding project was provided by Udacity.

Blog : https://medium.com/@trandangtin91/starbucks-capstone-challenge-50e0b61682a8
Source: [Udacity Nano Program: Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025?utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=12908932988_c_individuals&utm_term=124509203711&utm_keyword=%2Budacity%20%2Bdata%20%2Bscience_b&gclid=Cj0KCQjwxtSSBhDYARIsAEn0thQ37yvP0P4SRAW7XaiasAdiTOYFe-IfkrDUAbPxQNuZ_05CUs6ukj0aAlT-EALw_wcB)
