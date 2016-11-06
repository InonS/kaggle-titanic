# coding=utf-8
import csv
from os import cpu_count

import numpy as np
import pandas as pd
from numpy import zeros
from sklearn.ensemble import RandomForestClassifier


def process(data_file_path):
    df = pd.read_csv(data_file_path, header=0)

    print("information schema description:")
    information_schema_description(df)

    print("Summary of attributes with values missing:")
    missing_attributes_summary(df)

    print("Enumerating str/object data:")
    df = enumeration(df)

    print("Filling missing data:")
    df = fill(df)

    print("Separating labels and observation IDs from attributes")
    df, passenger_ids, y = separate_labels_and_ids_from_attribs(df)

    print("Feature engineering:")
    feature_engineering(df)

    print("Cleansed %s:" % data_file_path)
    check_cleansed(df)
    X = df.values

    print("Topology: ids = {}, features = {}, labels = {}".format(passenger_ids.shape, X.shape,
                                                                  None if y is None else y.shape))
    return passenger_ids, X, y


def separate_labels_and_ids_from_attribs(df):
    df, y = seperate_labels(df)
    df, passenger_ids = seperate_ids(df)

    print("don't even bother enumerating these fields, just drop them:")
    df = drop_extra_fields(df)

    return df, passenger_ids, y


def drop_extra_fields(df):
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    return df


def seperate_ids(df):
    passenger_ids = df['PassengerId'].values
    df = df.drop(['PassengerId'], axis=1)
    return df, passenger_ids


def seperate_labels(df):
    y = None
    if 'Survived' in df:
        print("Separating Survived, which is not part of the parameters (\"X\")")
        y = df['Survived']
        df = df.drop(['Survived'], axis=1)
    return df, y


def check_cleansed(df):
    information_schema_description(df)
    assert len(
        df.count().unique()) == 1, "not all attributes have the same number of observations! check for missing values"


def feature_engineering(df):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass


def fill(df):
    print("Filling other missing numeric values with zeroeth-order approximation -- the most probable value (mode):")
    fill_zeroeth_order(df)

    print("Filling Age:")
    df = fill_age(df)

    return df


def fill_age(df):
    nGenders = len(df['Gender'].dropna().unique())
    nPclasses = len(df['Pclass'].dropna().unique())
    median_ages = zeros((nGenders, nPclasses))
    for i in range(nGenders):
        for j in range(nPclasses):
            median_ages[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()
    df['AgeFill'] = df['Age']
    for i in range(nGenders):
        for j in range(nPclasses):
            df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j + 1), 'AgeFill'] = median_ages[i, j]
    df = df.drop(['Age'], axis=1)
    return df


def fill_zeroeth_order(df):
    for field in ('EmbarkedEnum', 'SibSp', 'Parch', 'Fare'):
        df[field].fillna(df[field].dropna().mode().values[0], inplace=True)


def enumeration(df):
    print("Enumerating Sex:")
    df = enumerate_sex(df)

    print("Enumerating Embarked:")
    df = enumerate_embarked(df)

    return df


def enumerate_embarked(df):
    print("unique 'Embarked' values: {}".format(df['Embarked'].unique()))
    df['EmbarkedEnum'] = df['Embarked'].dropna().map({'C': 1, 'S': 2, 'Q': 3}).astype(int)
    df = df.drop(['Embarked'], axis=1)
    return df


def enumerate_sex(df):
    print("unique Sex values: {}".format(df['Sex'].unique()))
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df = df.drop(['Sex'], axis=1)
    return df


def missing_attributes_summary(df):
    for i, s in enumerate(df):
        if df[s].count() < len(df):
            print("{} missing values in {} attribute".format(len(df) - df[s].count(), s))


def information_schema_description(df):
    print("info: {}".format(df.info()))
    print("describe: {}".format(df.describe()))


def train():
    model = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=cpu_count() - 2, verbose=3)

    ids, X, y = process('data/train.csv')
    model = model.fit(X, y)
    prediction = model_prediction(model, X)

    training_score = model.score(X, y)
    print(training_score)
    training_score = model.oob_score_
    print("training score = ", training_score)
    return model


def test(model):
    """
    Test, save public test as csv: 'PassengerId', 'Survived'
    """

    ids, X, y = process('data/test.csv')
    prediction = model_prediction(model, X)

    with(open('out/survival.csv', 'w')) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["PassengerId", "Survived"])
        csv_writer.writerows(zip(ids, prediction))


def model_prediction(model, X):
    prediction = model.predict(X)
    print("prediction shape = {}".format(prediction.shape))
    print("unique predictions = {}".format(np.unique(prediction)))
    print("prediction count = {}".format(np.bincount(prediction, minlength=2)))
    return prediction


if __name__ == '__main__':
    model = train()
    test(model)
