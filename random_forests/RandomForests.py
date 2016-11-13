# coding=utf-8
from csv import writer as csv_writer_
from datetime import datetime
from logging import basicConfig, debug, warning, info, DEBUG
from os import cpu_count

from numpy import zeros, unique, bincount, nan
from pandas import qcut
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
        a) Imputation of missing values (zero, mean, median, random draw from the hot deck distribution, k-nn, etc. based on dataset statistics)
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

        self.unique_titles = {}

    def transform(self):

        self.descriptive_feature_engineering()

        info("Filling missing data:")
        self.imputation()

        self.data_leak_example()

        info("Separating labels and observation IDs from kept attributes")
        passenger_ids, y = self.separate_id_and_target_from_features()

        info("Feature engineering:")
        self.predictive_feature_engineering()

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

    def predictive_feature_engineering(self):
        self.df['IsAdult'] = self.df['AgeFill'] > 16
        self.df['AgeBin'] = qcut(self.df['AgeFill'], 5)
        self.family_size()
        self.df[
            'Age*Class'] = self.df.AgeFill * self.df.Pclass  # survival more likely for young affluent individuals than elderly poor ones?

    def imputation(self):
        """
        TODO Multiple imputation?
        """

        info("Filling other missing numeric values with zeroeth-order approximation -- the most probable value (mode):")
        self.zeroeth_order_numerical_imputation()

        info("Filling Age using joint distribution:")
        # TODO EDA to find correlations. Perform hot deck imputation from these joint distribution.
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

        self.enumerate_name()

        self.enumerate_fare()

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

    def enumerate_name(self):
        self.enumerate_title()
        self.enumerate_family()

    def enumerate_fare(self):
        """
        TODO correlates with Pclass? assist in imputation of that field
        """
        self.df['FareBin'] = qcut(self.df['Fare'], 7)

    def tokenization(self):
        # "family_name": str
        self.tokenize_title()

    def family_size(self):
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch']
        self.df['HasNoFamily'] = self.df['SibSp'] + self.df['Parch'] == 0
        self.df['IsWithChildren'] = self.df['Parch'] > 0 and self.df['IsAdult'] > 0

    def data_leak_example(self):
        # info("extract binary features from 'SibSp' and 'ParCh' fields:")
        # "has_any_family_survived": bool
        # "is_perishing_mother": bool
        # "is_surviving_father": bool
        pass

    def descriptive_feature_engineering(self):
        """
        Develop features which correlate with others which are missing entries, and which will serve in their imputation
        If the original features add noise but no information (relative to the derived features), drop the original
        """

        info("Tokenize str data:")
        self.tokenization()

        info("Enumerating str/object data:")
        self.enumeration()

    def enumerate_title(self):
        """
        correlates with gender and age? assist in imputation of those fields
        TODO hard code equivalent titles to merge bins
        """
        self.df['TitleBin'] = self.df['Title'].dropna()
        self.df['TitleBin'] = self.df['TitleBin'].map(self.unique_titles).astype(int8)

    def enumerate_family(self):
        # "family_id": int
        pass

    def tokenize_title(self):
        """
        TODO suffix titles (e.g. MD, Ph.D., Esq.)
        """

        last_title_enum = 0

        for row in self.df.iterrows():

            name = row['Name']
            if name is None:
                continue

            name_str = str(name)
            if len(name_str) == 0:
                continue

            name_prefix = name_str.split(sep=' ', maxsplit=1)[0]
            if len(name_prefix) < 3 or name_prefix[-1] != '.':
                # prefix is not a title (probably first initial)
                continue

            title = name_prefix[:-2].capitalize()
            row['Title'] = title

            existing_title_enum = self.unique_titles[title]
            if existing_title_enum is None:
                last_title_enum += 1
                self.unique_titles[title] = last_title_enum


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

    with(
            open('out/survival_' + str(epoch_seconds_now()) + "_" + type(model).__name__ + '.csv',
                 'w')) as submission_file:
        csv_writer = csv_writer_(submission_file)
        csv_writer.writerow(["PassengerId", "Survived"])
        csv_writer.writerows(zip(ids, y_hat))


def epoch_seconds_now():
    return int(datetime.now().timestamp())


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

        y_hat = cross_val_predict(model, X, y, n_jobs=cpu_count() - 2, verbose=3)
    return y_hat


def model_selection():
    classifiers = {
        # linear models
        "lrcv": LogisticRegressionCV(max_iter=1000, n_jobs=cpu_count() - 2, verbose=3),
        "sgdc": SGDClassifier(verbose=3, n_jobs=cpu_count() - 2),
        "rccv": RidgeClassifierCV(),

        # naive Bayes
        "gnb": GaussianNB(),

        # ensemble (random forests, boosting, etc.)
        "rfc": RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=cpu_count() - 2, verbose=3),
        "xtc": ExtraTreesClassifier(n_estimators=100, bootstrap=True, oob_score=True, n_jobs=cpu_count() - 2,
                                    verbose=3),
        "gbc": GradientBoostingClassifier(verbose=3),

        # neural nets
        "mlpc": MLPClassifier(verbose=True)
    }

    independant_classifiers(classifiers)
    voting_classification(classifiers)


def independant_classifiers(classifiers):
    for model_name in classifiers.keys():
        model = classifiers[model_name]
        info("%s training %s" % (model_name, model))
        model = train(model)
        test(model)


def voting_classification(classifiers):
    info("training VotingClassifier")
    # models_by_name = [(model_name, classifiers[model_name]) for model_name in classifiers.keys()]
    voting = VotingClassifier(classifiers.items(), n_jobs=cpu_count() - 2)
    model = train(voting)
    test(model)


def setup_logger():
    log_file_name = __file__[:-3] + str(epoch_seconds_now()) + ".log"
    logger_stream_ = open(log_file_name, "a")
    basicConfig(level=DEBUG, stream=logger_stream_)
    return logger_stream_


if __name__ == '__main__':
    """
    TODO Pipeline
    """

    logger_stream = setup_logger()
    stdout = logger_stream
    stderr = logger_stream
    model_selection()
