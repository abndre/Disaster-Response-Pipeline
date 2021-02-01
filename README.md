# Disaster Response Pipeline Project

This is a projetct is a END-TO-END pipeline to creat a machine learning project
with a front-end interface to used.



### Instructions:

## 1. Virtualenv

create virtualenv
```
python3 -m venv .env
source .env/bin/activate
```

install requirements

```
pip install -r requirements.txt
```


## 2. Run projetct

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/