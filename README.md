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

All OK! So we crack on diving into the data.

## 1. Are developers able to achieve salary progression through years of service?
 * What is the correlation to their demographics?

In order to analyse if we observe correlations between salary compensation and developer experience considering the numbers of years they've been programming, we focus on the data features that provide such information.

`ConvertedComp` is the annual compensation value adapted to $ USD. This value comes directly from the StackOverflow dataset.

Of course, we want to focus on the responses where we have all the data points present, so we eliminate all entries that do not have one or more of data points we want to analyse:

```
df = df.dropna(axis=0, how="any", subset=['Age', 'Country', 'ConvertedComp', 'YearsCode', 'YearsCodePro'])
df
```

This leaves us with **48350** which is a good enough dataset volume to dissect.

We start by cleaning the data from most of its data points and keep the responses from professional developers (StackOverflow has data from students and hobyists), and by ensuring all data points are workable with. This means changing `String` values to concrete `numerical` values, removing `null` values and so forth.

Since compensation was an open field on the survey, we have quite diverse results down to the actual dollar.
Therefore, we aggregate compensation packages in ranges so we understand the distribution of responses existing in our dataset:

```
(...)
possible_values = ["Below $40k", "$40k - $80k", "$80k - $120k", "$120k - $160k", "$160k - max"];
(...)
(compensation_values).plot(kind="bar", legend=None); # / df.ConvertedComp.count()
plt.title("Compensation distribution");
plt.xticks([0, 1, 2, 3, 4], possible_values);
```
![Number of responses per compensation ranges](https://github.com/Halflernation/data-science-project1/images/1.2.png "# responses per compensation ranges")

Using SeaBorn library we can easily get a heatmap of correlations between the data points in analysis:
![Heatmap of correlation](https://github.com/Halflernation/data-science-project1/images/1.3.png "Heatmap of correlation")

This is a real kick in the teeth!

What we wanted to check for a potential correlation shows the lowest correlation value. We're not able to work more on this since the data is so scattered that we cannot make predictions or show correlations between salary compensation and years of experience coding.

### What are the implications for this lack of correlation?
 * respondents are not telling the whole truth
 * We have an unrepresentative data values for the actual developers in existence (skewed dataset)
 * Salary progression is not present in developers careers
 * Salary offers to programmers is not directly correlated to their years of experience
 * We're looking this at a macro level (globally) so we're trying to correlate compensation values that are relevant locally, but showcase a gap to other geographies and economies.

We have to use the data assuming this is a valuable and honest representation of reality, and we know from economy figures that developers are one of the most sought after skills in today's world. Therefore, what may be impacting our results might be the fact that we're using the data too much aggregated.

We quickly check this by taking the `United Kingdom` (a country that is holds a key spot in being a tech hub at global stage) as a sample and plot the same heatmap.

![UK heatmap of correlation](https://github.com/Halflernation/data-science-project1/images/1.4.png "UK heatmap of correlation")

Unfortunately we observe no real change. This means StackOverflow data does not allow us to make conclusions on compensation packages based on developer years of experience.

### Hidden impacts

Trying to find out why this is the case, it can quickly be observed in the dataset the actual data present on the dataset for compensation values:

```
df_real.sort_values(by=['YearsCode'])
```

![UK heatmap of correlation](https://github.com/Halflernation/data-science-project1/images/1.5.png "UK heatmap of correlation")

One respondent from Spain has put his annual compensation as being the equivalent to $ 29 USD.

This isn't right! StackOverflow does not seem to perform any checks on the data nor clean it up.

In this istance, one can assume the respondent could be referring to a compensation of $ 29 000 USD / year, but it is truly unclear if that is the case.

Much more work is required to filter and clean the data points and perform correlation and prediction to local geographies or roughly equal economies. A compensation salary in a developing country is not the same as in a developed Western country.


## 2. Where are the cheapest developers? (lowest average salary)

From the same data point analysis, we can quickly derive which country shows the lowest mean salary converted to $ USD.

```
(...)
res = df['Country'].value_counts()

res = df.groupby(['Country']).ConvertedComp.mean().sort_values()
res
```

![asdadasd](https://github.com/Halflernation/data-science-project1/images/2.1.png "adsadsadsadsad")

Overall, Guinea shows the lowest median annual compensation package when local currency is converted to $ USD.

This needne't mean it is the worst country for developers, as of course, local currency significance will dictate how real compensation packages are adapted to local economy.

## 3. How well can we predict Salary based on Country?

As observed in #2, compensation packages vary greatly from country to country. But can we predict a typical salary (or range) for a developer on a per-country basis?

To respond to this question we need to analyse the data at a macro level again. From the previous `business questions` we already have the data available from our dataset (despite disclaimers on its veracity or variance), all we need to do is to convert the `Country` data information onto computable information for our model.

In order to do that we use `Pandas.get_dummies` to convert `String` categorical data onto a binary indicators:

```
d = pd.get_dummies(data=df, columns=['Country'], drop_first=False, prefix='', prefix_sep='')
d = d[columns]
d
```

![Country binary indicator values](https://github.com/Halflernation/data-science-project1/images/3.1.png "Country binary value indicator")

Therefore, we just need to use the data to build a model to predict the computed compensation salary for a given country.

```
countries = df.Country.unique()

X = d[countries]
y = d['ConvertedComp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

lm_model = LinearRegression(normalize=True) 

y_test_preds = lm_model.predict(X_test) # Predictions here using X_train,y_train on lm_model
r2_test =  r2_score(y_test, y_test_preds) # Rsquared here for comparing test and predictions from lm_model
r2_test
```

Output: `-1.5807`

### Follow-on from our first business question

This may feel intuitive since within each geopgraphy there will be plenty of developers and each at different stages of their careers (thus expertise). But analysing the data further as we did before, one observes developers starting out their career on the first professional job with extremely high compensation packages (max salary reported: $ 200k USD/year is also reported from a respondent with 1 year of experience).

Naturally, this makes insight difficult to extract from the data.

# What can be done?

It is required to drill down the data further and extract the maximum of insight it provides to us:

 * One can further drill down and understand where we verify that extremely high paid jobs are given to (potentially) innexperient developers
 * Understand how skill specialisation translates to the effect of the previous point
 * Observe which other parameters shape this variance

It is fair to keep an open mind to each dataset a data scientist uses, specially to its state and nature of how the data is gathering.
