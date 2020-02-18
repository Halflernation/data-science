# Can You Rely on StackOverflow Developer Survey Data?

## StackOverflow 2019 data analysis

In this blog post we continue to dissect information from the StackOverflow 2019 survey to see if we can extract further insight from how developers are compensated.

Our main focus in order to create our business questions is what impacts compensation: i) developer experience and ii) geography.
Of course many other parameters influence, perhaps just as much the actual compensation package. But we'll leave that to further analysis later on.

Our business questions:
 1. Are developers able to achieve salary progression through years of service?
    - what is the correlation to their demographics?
 2. Where are the cheapest developers? (lowest average salary)
 3. How well can we predict Salary based on Country?

With our questions in mind, the first step is to analyse the data and clean it so we use the relevant data features in order to answer the questions.

## Exploring the dataset

Our dataset comes from StackOverflow 2019 Survey results: https://insights.stackoverflow.com/survey

We ran a sample to understand the data and make sure the data looks ok:

```
df = pd.read_csv('survey_results_public.csv')
df
```
![DataFrame as loaded in Pandas](https://github.com/Halflernation/data-science-project1/images/1.1.png "DataFrame as loaded in Pandas")


