# Statistical Data Preprocessing

A python package to create a data cleaning and pre-processing pipe line. 

## Where to use

1. When you have data in matrix form where each row is a datapoint and each column is a feature

2. You need to perform data cleaning tasks such as following in one function call:
  * Deal with nans, nulls and None values (i.e. remove or replace nans)
  * Deal with infs, -inf (i.e remove infs, or use appropriate scheme to substitute infs)
  * Remove columns with just one unique value (Essentialy useless columns)
  * Given a dirty column with mixed data types such as ints and strings in the same column, split it in to two columns
  * Normalize data using one of standard or robust scaling schemes
  * Merge values a categorical feature which occur less then a threshold
  * Delete or replace outliers

3. You need to perform feature engineering tasks such as following in one function call:
  * Create a boolean column of is_nan to retain information if a feature had nans, before you get rid of nans
  * Create a column of is_inf to retain information if a feature had inf or ninf (-inf), before you get rid of infs
  * Extract various features from a datetime column
  * One hot encode categorical a variable
  * Create a new column to mark outliers against a singe feature 

4. Perform any of the above mention tasks in a batch for each column of a dataframe

5. Create a piple line to automate clean data creation from raw data.

6. Perform any of above operations optimally

## Install requirements
```
pip install -r requirements.txt
```

## How to use

Please refer to example.ipynd inside the main branch to have a look at the working example which demonstrates the use for this package.



