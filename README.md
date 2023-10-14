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

Blog : https://medium.com/@trandangtin91/starbucks-capstone-challenge-50e0b61682a8
Source: [Udacity Nano Program: Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025?utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=12908932988_c_individuals&utm_term=124509203711&utm_keyword=%2Budacity%20%2Bdata%20%2Bscience_b&gclid=Cj0KCQjwxtSSBhDYARIsAEn0thQ37yvP0P4SRAW7XaiasAdiTOYFe-IfkrDUAbPxQNuZ_05CUs6ukj0aAlT-EALw_wcB)
