# coding=utf-8
from csv import writer as csv_writer_
from logging import basicConfig, debug, warning, info, DEBUG
from os import cpu_count

from numpy import zeros, unique, bincount
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict


class DataProcessor:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path

        self.df = read_csv(self.data_file_path, header=0)

        info("information schema description:")
        self.information_schema_description()

        info("Summary of attributes with values missing:")
        self.missing_attributes_summary()

    def __call__(self, *args, **kwargs):

        info("Enumerating str/object data:")
        self.enumeration()

        info("Filling missing data:")
        self.imputation()

        info("Separating labels and observation IDs from attributes")
        passenger_ids, y = self.separate_labels_and_ids_from_attribs()

        info("Feature engineering:")
        self.feature_engineering()

        info("Cleansed %s:" % self.data_file_path)
        self.check_cleansed()
        X = self.df.values

        info("Topology: ids = {}, data samples = {}, labels = {}".format(passenger_ids.shape, X.shape,
                                                                         None if y is None else y.shape))
        self.passenger_ids = passenger_ids
        self.X = X
        self.y = y

        return self.passenger_ids, self.X, self.y

    def separate_labels_and_ids_from_attribs(self):
        y = self.seperate_labels()
        ids = self.seperate_ids()

        debug("don't even bother enumerating these fields, just drop them:")
        self.drop_extra_fields()

        return ids, y

    def drop_extra_fields(self):
        self.df = self.df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    def seperate_ids(self):
        ids = self.df['PassengerId'].values
        self.df = self.df.drop(['PassengerId'], axis=1)
        return ids

    def seperate_labels(self):
        y = None
        if 'Survived' in self.df:
            debug("Separating Survived, which is not part of the parameters (\"X\")")
            y = self.df['Survived']
            self.df = self.df.drop(['Survived'], axis=1)
        return y

    def check_cleansed(self):
        self.information_schema_description()
        assert len(
            self.df.count().unique()) == 1, \
            "not all attributes have the same number of observations! check for missing values"

    def feature_engineering(self):
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch']
        self.df['Age*Class'] = self.df.AgeFill * self.df.Pclass

    def imputation(self):
        info("Filling other missing numeric values with zeroeth-order approximation -- the most probable value (mode):")
        self.zeroeth_order_numerical_imputation()

        info("Filling Age:")
        self.impute_age()

    def impute_age(self):
        n_genders = len(self.df['Gender'].dropna().unique())
        n_pclasses = len(self.df['Pclass'].dropna().unique())
        median_ages = zeros((n_genders, n_pclasses))
        for i in range(n_genders):
            for j in range(n_pclasses):
                median_ages[i, j] = self.df[(self.df['Gender'] == i) & (self.df['Pclass'] == j + 1)][
                    'Age'].dropna().median()
        self.df['AgeFill'] = self.df['Age']
        for i in range(n_genders):
            for j in range(n_pclasses):
                self.df.loc[(self.df.Age.isnull()) & (self.df.Gender == i) & (self.df.Pclass == j + 1), 'AgeFill'] = \
                    median_ages[i, j]
        self.df = self.df.drop(['Age'], axis=1)

    def zeroeth_order_numerical_imputation(self):
        for field in ('EmbarkedEnum', 'SibSp', 'Parch', 'Fare'):
            self.df[field].fillna(self.df[field].dropna().mode().values[0], inplace=True)

    def enumeration(self):
        debug("Enumerating Sex:")
        self.enumerate_sex()

        debug("Enumerating Embarked:")
        self.enumerate_embarked()

    def enumerate_embarked(self):
        debug("unique 'Embarked' values: {}".format(self.df['Embarked'].unique()))
        self.df['EmbarkedEnum'] = self.df['Embarked'].dropna().map({'C': 1, 'S': 2, 'Q': 3}).astype(int)
        self.df = self.df.drop(['Embarked'], axis=1)

    def enumerate_sex(self):
        debug("unique Sex values: {}".format(self.df['Sex'].unique()))
        self.df['Gender'] = self.df['Sex'].map({'female': 0, 'male': 1}).astype(int)
        self.df = self.df.drop(['Sex'], axis=1)

    def missing_attributes_summary(self):
        for i, s in enumerate(self.df):
            if self.df[s].count() < len(self.df):
                warning("%d missing values in %s attribute" % (len(self.df) - self.df[s].count(), s))

    def information_schema_description(self):
        debug("head: {}".format(self.df.head()))
        debug("tail: {}".format(self.df.tail()))
        debug("info: {}".format(self.df.info()))
        debug("describe: {}".format(self.df.describe()))


def train(model):
    processor = DataProcessor('data/train.csv')
    ids, X, y = processor()

    model = model.fit(X, y)
    model_prediction(model, X, y)
    return model


def test(model):
    """
    Test, save public test as csv: 'PassengerId', 'Survived'
    """

    processor = DataProcessor('data/test.csv')
    ids, X, y = processor()

    prediction = model_prediction(model, X, y)

    with(open('out/survival.csv', 'w')) as f:
        csv_writer = csv_writer_(f)
        csv_writer.writerow(["PassengerId", "Survived"])
        csv_writer.writerows(zip(ids, prediction))


def model_prediction(model, X, y=None):
    prediction = model.predict(X)
    debug("prediction shape = {}".format(prediction.shape))
    debug("unique predictions = {}".format(unique(prediction)))
    debug("prediction count = {}".format(bincount(prediction, minlength=2)))
    if y is not None:
        score = model.score(X, y)
        debug("score = %f" % score)

        scores = cross_val_score(model, X, y, n_jobs=cpu_count() - 2, verbose=3)
        debug("cross-validation scores: {}".format(scores))  # [ 0.78787879  0.83164983  0.81144781]

        prediction = cross_val_predict(model, X, y, n_jobs=cpu_count() - 2, verbose=3)
    return prediction


if __name__ == '__main__':
    basicConfig(level=DEBUG)
    # TODO Pipeline
    model_ = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=cpu_count() - 2, verbose=3)
    model_ = train(model_)
    test(model_)
