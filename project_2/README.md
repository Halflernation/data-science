# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
    - Output:
        
```   
Evaluating model...
related                   0.734210
request                   0.778442
offer                     0.998805
aid_related               0.754931
medical_help              0.941622
medical_products          0.966129
search_and_rescue         0.977286
security                  0.988045
military                  0.994820
child_alone               1.000000
water                     0.935246
food                      0.883244
shelter                   0.900379
clothing                  0.988643
money                     0.988444
missing_people            0.992827
refugees                  0.982666
death                     0.974098
other_aid                 0.834429
infrastructure_related    0.964933
transport                 0.984061
buildings                 0.965531
electricity               0.993624
tools                     0.997011
hospitals                 0.994023
shops                     0.997211
aid_centers               0.991831
other_infrastructure      0.981470
weather_related           0.877067
floods                    0.976689
storm                     0.977286
fire                      0.995816
earthquake                0.935645
cold                      0.993624
other_weather             0.981072
direct_report             0.769874

Best Parameters: {'features__text_pipeline__tfidf__smooth_idf': True, 'features__text_pipeline__tfidf__sublinear_tf': True}
Labels:
['0' '1']
```

No classification output is generated due to `multiclass-multioutput is not supported` error when using sklearn standard functions.
(labels are purely 0's and 1's: 

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
