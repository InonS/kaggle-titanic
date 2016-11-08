# coding=utf-8
from csv import writer as csv_writer_
from datetime import datetime
from logging import basicConfig, debug, warning, info, DEBUG
from os import cpu_count

from numpy import zeros, unique, bincount, nan
from pandas import read_csv, DataFrame
from pandas.algos import int8
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict


class DataProcessor:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path

        self.df = DataFrame(read_csv(self.data_file_path, header=0))  # cast as DataFrame to disambiguate type inference

        info("information schema description:")
        self.information_schema_description()

        info("Summary of attributes with values missing:")
        self.missing_attributes_summary()

    def __call__(self, *args, **kwargs):

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

    # confusion_matrix(y[1], prediction[1])

    epoch_time = int(datetime.now().timestamp())
    with(open('out/survival_' + str(epoch_time) + '.csv', 'w')) as f:
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
    """
    TODO Pipeline
    """
    basicConfig(level=DEBUG)

    # rfc = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=cpu_count() - 2, verbose=3)
    # xtc = ExtraTreesClassifier(n_estimators=100, bootstrap=True, oob_score=True, n_jobs=cpu_count() - 2, verbose=3)
    # gbc = GradientBoostingClassifier(verbose=3)
    # mlpc = MLPClassifier(verbose=True)
    # estimators = enumerate((rfc, xtc, gbc, mlpc))
    #
    # for i, model_ in estimators:
    #     info("%d training %s" % (i, str(model_)))
    #     model_ = train(model_)
    #     test(model_)
    #
    # voting = VotingClassifier([model_ for model_ in estimators], n_jobs=cpu_count() - 2)
    # info("training VotingClassifier")
    # model_ = train(voting)
    # test(model_)

    model_ = GradientBoostingClassifier(verbose=3)
    model_ = train(model_)
    test(model_)
