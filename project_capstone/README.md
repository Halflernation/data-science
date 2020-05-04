# Project Capstone

This content is part of Udacity Data Science Nano-degree Final Project Capstone.


### Instructions:
1. Download all data from project_capstone folder

2. Unzip minified version of data

3. Run Jupyter Notebook

    - Run Jupyter notebook.


### Overview

This project works on a tiny subset (128 MB) of the full dataset available (12 GB).

The data refers to a music streaming service and its users' interaction with it.
The exploration and explanation of fields in this dataset is not a part of this project,
henceforth it is advisable to those not familiar with this dataset to first explore themselves.

We call the streaming service: Sparkify.


### Problem Statement

In any subscription-based service there's always movement insofar how users are engaged with the 
service, how they stay subscribed, or change/cancel their subscription.

It is of the utmost important to service prodivers to understand why their users may cancel or 
reduce their subscription service. Since this impacts how the service is monetised, one of the 
key industry metrics is to define: `churn` as the users who cancel/downgrade service subscription
over the total amount of currently subscribed users.

A low `churn` value means the service has `stickyness`, meaning, users who subscribe to the service
stay connected and continue using it, and most importantly: paying for it.

Understanding or being able to predict `churn` values based on user's interactions with the platform 
may give great insight to a service provider to:

	- What leads a user to `churn`
	
	- Segmenting user's behaviour to identify potential `churn` in advance
	(this can be useful when trying to keep users engaged and subscribed, with offers, or other promotional content)

If we're able to correctly predict if a given user would /would not cancel/downgrade their subscription
then we are able to take action before `churn` takes place.

This project analyses the dataset provided in the #Overview and makes an accurate test prediction based 
on how many songs the users listen to and how many songs their sessions tend to have.


### Evaluation

We evaluate our results against a subset of the `data` where we classify users who `have` churned 
and users who `have not`. This is essentially a label attached to an individual user, which we'll
use to later check if the prediction was accurate or not.

We train a Linear Regression and a Logistic Regression model with a split of the `data` as 70%/30%: 
training / test from the original `data` subset.

Unfortunately, there's not many users who we'd want to evaluate under the conditions that they have used the
service and listened to music sequentially, distributed between those who continue to use the service and those
who have cancelled / downgraded it.

Therefore we'll use F1 score to determine the model's accuracy.



### Results

We build two models using Linear and Logistic regression to predict `churn`.

The Linear Regression model does not work as expected and is not able to pick-up. This is due to the non-linear
relationship between the features.

On the other hand, we're able to predict successfully with the Logistic Regression model with an accuracy
around ~ 90%.

It is true the sample data is really small and this should be having an impact on the overall accuracy 
and definition of features and how they're truly related to one another. But even changing a few 
of the models' parameters makes the prediction to still be > 85%.

The models parameters output do specify a rough accuracy of 0.838509 (~ 84%), therefore it is assumed
that with a larger dataset the accuracy would converge to that value.

Again, it is not a great result in terms of model accuracy, but indeed it is an usable model in this scenario
where even if the service provider is able to get an indication of a potential change of heart for a given
user to change / cancel their subscription and take preemptive action on that prediction.


### Tuning

A cross-validation is made to ensure the metrics originally used are adequate or if there are
substantial gains by changing Logistic model's parameters.

With `maxIter` values of:

'''
[5, 10, 15, 25]
'''

And `regParam` values of:

'''
[0.0, 0.5, 1.0]
'''

The test results transform under the testing dataset further predict with a ~84% rate.
Another showcase the results are easily affected due to the very small sampling size.


### Overall

The performance of the model is reasonable and more steps are required to identify the features that
affect `churn` on the Sparkify service.

The current `churn` prediction is valuable and one shouldn't be shy to seek improvements.

# License

This project is licensed under the MIT License:

Copyright 2020 Halflernation

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.