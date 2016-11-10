# coding=utf-8
from csv import writer as csv_writer_
from datetime import datetime
from logging import basicConfig, debug, warning, info, DEBUG
from os import cpu_count

from numpy import zeros, unique, bincount, nan
from pandas import read_csv, DataFrame
from pandas.algos import int8
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class DataProcessor:
    """
    TODO inherit sklearn.base.TransformerMixin

    1) Enrich features by bringing in more sources
    2) Remove any known background biases
    3) Encoding (a.k.a. enumeration) of categorical features
    4) Vectorization: Break up data object into numeric components, represented as a vector
    5) Vector Normalization: Scaling vector quantities to have unit magnitude (if only direction / relative component weights are important)
    6) Statistically consistent procedures (should be preformed prior to any inconsistent ones)
        a) Standardization (a.k.a. z-score): Shifting to zero mean, and scaling to unit variance
    7) Statistically inconsistent procedures
        a) Imputation of missing values (zero, mean, median, k-nn, etc. based on dataset statistics)
        b) Sub-sampling: Censoring, Binning (the limit of which is Binarization)
    8) Enrich features (e.g. generating polynomial features)
    """

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path

        self.df = DataFrame(read_csv(self.data_file_path, header=0))  # cast as DataFrame to disambiguate type inference

        info("information schema description:")
        self.information_schema_description()

        info("Summary of attributes with values missing:")
        self.missing_attributes_summary()

    def transform(self):

        info("Enumerating str/object data:")
        self.enumeration()

        info("Filling missing data:")
        self.imputation()

        info("Separating labels and observation IDs from kept attributes")
        passenger_ids, y = self.separate_id_and_target_from_features()

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

    def separate_id_and_target_from_features(self):
        ids = self.seperate_ids()
        y = self.seperate_labels()

        debug("don't even bother enumerating these fields, just drop them:")
        self.drop_extra_fields()

        return ids, y

    def drop_extra_fields(self):
        self.df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    def seperate_ids(self):
        ids = self.df['PassengerId'].values
        self.df.drop(['PassengerId'], axis=1, inplace=True)
        return ids

    def seperate_labels(self):
        y = None
        if 'Survived' in self.df:
            debug("Separating Survived, which is not part of the parameters (\"X\")")
            y = self.df['Survived']
            self.df.drop(['Survived'], axis=1, inplace=True)
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
        """
        TODO Multiple imputation?
        """

        info("Filling other missing numeric values with zeroeth-order approximation -- the most probable value (mode):")
        self.zeroeth_order_numerical_imputation()

        info("Filling Age:")
        self.impute_age()

    def impute_age(self):
        """
        Collect median Age for each Passenger segment, based on Gender and Pclass (which don't have to be enumerated!).
        Impute Age for each Passenger missing an Age, based on median of other passengers with same Gender and Pclass.
        TODO: Generalize for *feature_names. Create generator taking *unique_values_for_feature and returning tuple of enumerations
        """
        # create empty bins for median Age, based on each possible unique combination of Gender and Pclass
        gender_values = self.df['Gender']
        unique_genders = list(gender_values.dropna().unique())
        pclass_values = self.df['Pclass']
        unique_pclasses = list(pclass_values.dropna().unique())
        median_ages = zeros((len(unique_genders), len(unique_pclasses)))

        # collect median Age for each Passenger segment, based on Gender and Pclass.
        for i, g in enumerate(unique_genders):
            for j, c in enumerate(unique_pclasses):
                entries_given_gender_and_pclass = (gender_values == g) & (pclass_values == c)
                median_ages[i, j] = self.df[entries_given_gender_and_pclass]['Age'].dropna().median()

        # start with existing ages, missing values will be overwritten
        self.df['AgeFill'] = self.df['Age']

        # Impute Age for each Passenger missing an Age, based on median of other passengers with same Gender and Pclass.
        for i, g in enumerate(unique_genders):
            for j, c in enumerate(unique_pclasses):
                entries_missing_age_with_given_gender_and_pclass = \
                    (self.df.Age.isnull()) & (self.df.Gender == g) & (self.df.Pclass == c)
                self.df.loc[entries_missing_age_with_given_gender_and_pclass, 'AgeFill'] = median_ages[i, j]

        # drop old column (with missing values)
        self.df.drop(['Age'], axis=1, inplace=True)

    def zeroeth_order_numerical_imputation(self):
        """
        This makes sense even for unsorted enumerations, however using the average does not.
        """
        for field in ('EmbarkedEnum', 'SibSp', 'Parch', 'Fare'):
            mode = self.df[field].dropna().mode().values[0]
            self.df[field].fillna(mode, inplace=True)

    def enumeration(self):
        debug("Enumerating Sex:")
        self.enumerate_sex()

        debug("Enumerating Embarked:")
        self.enumerate_embarked()

    def enumerate_embarked(self):
        embarked_values = self.df['Embarked']
        debug("unique 'Embarked' values: {}".format(embarked_values.unique()))
        self.df['EmbarkedEnum'] = embarked_values.dropna()
        self.df['EmbarkedEnum'] = self.df['EmbarkedEnum'].map({nan: 0, 'C': 1, 'S': 2, 'Q': 3}).astype(int8)
        self.df.drop(['Embarked'], axis=1, inplace=True)

    def enumerate_sex(self):
        debug("unique Sex values: {}".format(self.df['Sex'].unique()))
        self.df['Gender'] = self.df['Sex'].map({'female': 0, 'male': 1}).astype(int8)
        self.df.drop(['Sex'], axis=1, inplace=True)

    def enumerate_values(self, feature_name: str) -> None:
        """
        TODO enumerate using positive and negative integers symmetrically around zero,
        with more common values mapped to enumeration closer to zero.
        """
        feature_values = self.df[feature_name]
        unique_values = feature_values.unique()
        debug("unique " + feature_name + " values: {}".format(unique_values))
        unique_values_dict = {v: i for i, v in enumerate(unique_values)}
        self.df[feature_name + '_enum'] = feature_values.map(unique_values_dict).astype(int)

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
    ids, X, y = processor.transform()

    model = model.fit(X, y)
    y_hat = model_prediction(model, X, y)
    confusion_matrix(y, y_hat)
    classification_report(y, y_hat)
    return model


def test(model):
    """
    Test, save public test as csv: 'PassengerId', 'Survived'
    """

    processor = DataProcessor('data/test.csv')
    ids, X, y = processor.transform()

    y_hat = model_prediction(model, X)

    epoch_time = int(datetime.now().timestamp())
    with(open('out/survival_' + str(model) + "_" + str(epoch_time) + '.csv', 'w')) as submission_file:
        csv_writer = csv_writer_(submission_file)
        csv_writer.writerow(["PassengerId", "Survived"])
        csv_writer.writerows(zip(ids, y_hat))


def model_prediction(model, X, y=None):
    y_hat = model.predict(X)
    debug("prediction shape = {}".format(y_hat.shape))
    debug("unique predictions = {}".format(unique(y_hat)))
    debug("prediction count = {}".format(bincount(y_hat, minlength=2)))
    if y is not None:
        score = model.score(X, y)
        debug("score = %f" % score)

        scores = cross_val_score(model, X, y, n_jobs=cpu_count() - 2, verbose=3)
        debug("cross-validation scores: {}".format(scores))  # [ 0.78787879  0.83164983  0.81144781]
    else:
        y_hat = cross_val_predict(model, X, y, n_jobs=cpu_count() - 2, verbose=3)
    return y_hat


def model_selection():
    # linear models
    lrcv = LogisticRegressionCV(verbose=3, n_jobs=cpu_count() - 2)
    sgdc = SGDClassifier(verbose=3, n_jobs=cpu_count() - 2)
    rccv = RidgeClassifierCV()

    # naive Bayes
    gnb = GaussianNB()

    # ensemble (random forests, boosting, etc.)
    rfc = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=cpu_count() - 2, verbose=3)
    xtc = ExtraTreesClassifier(n_estimators=100, bootstrap=True, oob_score=True, n_jobs=cpu_count() - 2, verbose=3)
    gbc = GradientBoostingClassifier(verbose=3)

    # neural nets
    mlpc = MLPClassifier(verbose=True)

    classifiers = enumerate((lrcv, sgdc, rccv, gnb, rfc, xtc, gbc, mlpc))

    independant_classifiers(classifiers)
    voting_classification(classifiers)


def independant_classifiers(classifiers):
    for i, model in classifiers:
        info("%d training %s" % (i, str(model)))
        model = train(model)
        test(model)


def voting_classification(classifiers):
    info("training VotingClassifier")
    voting = VotingClassifier([model for model in classifiers], n_jobs=cpu_count() - 2)
    model = train(voting)
    test(model)


if __name__ == '__main__':
    """
    TODO Pipeline
    """
    basicConfig(level=DEBUG)

    model_selection()

    model_ = GradientBoostingClassifier(verbose=3)
    model_ = train(model_)
    test(model_)