# Identify Fraud from Enron Email


## Install

This project requires Python 2.7 and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Scikit-learn](http://scikit-learn.org/stable/)

## Code

The code is provided in the notebook `identifyFraudFromEnronEmail.ipynb`.

To open it, go to the top-level project directory `identifyFraudFromEnronEmail/` and start the notebook server:

```jupyter notebook```

This should open a web browser to the server's dashboard (typically `http://127.0.0.1:8888`). Click on the appropriate notebook (`.ipynb`) to open it, and follow the instructions.

## Run

To run a code cell in the notebook, hit `Shift+Enter`. Any output will be displayed below the corresponding cell.

You can also add/edit markdown text cells and render them using `Shift+Enter`.

## Data

The dataset used in this project is store in a Python dictionary created by combining the Enron email and financial data, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)
