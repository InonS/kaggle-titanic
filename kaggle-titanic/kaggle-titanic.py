# coding=utf-8

"""
Kaggle Titanic
"""

from collections import namedtuple
from csv import writer as csv_writer_
from datetime import datetime
from logging import DEBUG, basicConfig, debug, info, warning
from os import cpu_count
from os.path import sep

from matplotlib.pyplot import savefig, tight_layout
from numpy import bincount, nan, unique, zeros
from pandas import DataFrame, qcut, read_csv
from pandas.algos import int8
from seaborn import jointplot
from seaborn.linearmodels import corrplot
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def epoch_seconds_now():
    """
    :return: whole number of seconds since midnight on New Year's night, 1970, until the current moment.
    :rtype: int
    """
    return int(datetime.now().timestamp())


out_file_no_path = '_'.join((str(__file__)[:-3].split(sep)[-1], str(epoch_seconds_now())))
plot_file_base = sep.join(("out", "plot", out_file_no_path))


class DataProcessor:
    """
    TODO inherit sklearn.base.TransformerMixin

    Pre-processing procedures:
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

    Implementation overview:
    A. Descriptive feature engineering:
        Features which can be assumed to correlate with at least one feature which has missing values.
    B. Imputation
    C. Predictive feature engineering:
        Features which can be assumed to correlate with the target variable.
    """

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path

        self.df = DataFrame(read_csv(self.data_file_path, header=0))  # cast as DataFrame to disambiguate type inference

        self.passenger_ids = None
        self.X = None
        self.y = None

        info("information schema description:")
        self.information_schema_description()
        tight_layout()
        savefig('_'.join((plot_file_base, 'init')))

        info("Summary of attributes with values missing:")
        self.missing_attributes_summary()

        self.unique_titles = {None: -1}
        self.unique_tokens = {}
        self.unique_families = {None: -1}

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
        tight_layout()
        savefig('_'.join((plot_file_base, 'cleansed')))
        assert len(
            self.df.count().unique()) == 1, \
            "not all attributes have the same number of observations! check for missing values"

    def predictive_feature_engineering(self):
        self.df['IsAdult'] = self.df['AgeFill'] > 16
        self.df['AgeBin'] = qcut(self.df['AgeFill'], 5, labels=False)
        self.family_size()
        self.df[
            'Age*Class'] = self.df.AgeFill * self.df.Pclass  # survival more likely for young affluent individuals than elderly poor ones?

    def imputation(self):
        """
        TODO Multiple imputation?
        """

        info("Filling other missing numeric values with zeroeth-order approximation -- the most probable value (mode):")
        self.zeroth_order_numerical_imputation()

        info("Filling Age using joint distribution:")
        # TODO EDA to find correlations. Perform hot deck imputation from these joint distribution.
        self.impute_age()

    def impute_age(self):
        """
        Collect median Age for each Passenger segment, based on Gender and Pclass (which don't have to be enumerated!).
        Impute Age for each Passenger missing an Age, based on median of other passengers with same Gender and Pclass.
        TODO: Generalize for *feature_names. Create generator taking *unique_values_for_feature and returning tuple of enumerations

        http://pandas.pydata.org/pandas-docs/stable/indexing.html#different-choices-for-indexing
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
                    (self.df.Age.isnull()) & (self.df.Gender == g) & (self.df.Pclass == c)  # binary indexing
                self.df.loc[entries_missing_age_with_given_gender_and_pclass, 'AgeFill'] = median_ages[i, j]

        # drop old column (with missing values)
        self.df.drop(['Age'], axis=1, inplace=True)

    def zeroth_order_numerical_imputation(self):
        """
        This makes sense even for unsorted enumerations, however using the average does not.
        """
        for field in ('EmbarkedEnum', 'SibSp', 'Parch', 'Fare', 'FareBin'):
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
        series = self.df[feature_name]
        modes = series.mode()
        debug("unique " + feature_name + " modes (sorted): {}".format(modes))

        def symmetrize_by_pairity(order):
            return order / 2 if order % 2 == 0 else int((order + 1) / -2)

        unique_values_dict = {mode: symmetrize_by_pairity(order) for order, mode in enumerate(modes)}
        self.df[feature_name + '_enum'] = series.map(unique_values_dict).astype(int)

    def missing_attributes_summary(self):
        for i, s in enumerate(self.df):
            if self.df[s].count() < len(self.df):
                warning("%d missing values in %s attribute" % (len(self.df) - self.df[s].count(), s))

    def information_schema_description(self):
        debug("head: {}".format(self.df.head()))
        debug("tail: {}".format(self.df.tail()))
        debug("info: {}".format(self.df.info()))
        debug("describe: {}".format(self.df.describe()))
        debug("covariance: {}".format(self.df.cov()))
        debug("correlation: {}".format(self.df.corr()))
        # TODO pairplot(self.df)
        corrplot(self.df)
        # mask_null = isnull(self.df)
        # values = self.df.values
        # mask_nan = isnan(values)
        # heatmap(self.df, mask=mask)

    def enumerate_name(self):
        self.enumerate_title()
        self.enumerate_family()

    def enumerate_fare(self):
        """
        TODO correlates with Pclass? assist in imputation of that field
        """
        self.df['FareBin'] = qcut(self.df['Fare'], 7, labels=False)
        debug("correlation between FareBin and Pclass (higher fare expected to correlate with higher class, \
        i.e. smaller Pclass) = {}".format(self.df.corr()['FareBin']['Pclass']))
        jointplot('FareBin', 'Pclass', self.df)
        tight_layout()
        savefig('_'.join((plot_file_base, 'farebin_vs_pclass')))

    def tokenization(self):
        self.tokenize_family()
        self.tokenize_title()

    def family_size(self):
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch']
        self.df['HasNoFamily'] = self.df['SibSp'] + self.df['Parch'] == 0
        self.is_with_children()

    def is_with_children(self):
        self.df.insert(self.df.ndim, 'IsWithChildren', False)
        is_adult_with_children_or_parents = (self.df['Parch'] > 0) & (self.df['IsAdult'] > 0)  # binary indexing
        self.df.loc[is_adult_with_children_or_parents, 'IsWithChildren'] = True

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
        TODO hard code equivalent titles to merge bins: {'Mme': 8, 'Miss': 3, 'Capt': 15, 'Rev': 6, 'Don': 5, 'Dr': 7, \
        'Ms': 9, 'Lady': 11, 'Mr': 1, 'Mrs': 2, 'Mlle': 13, 'Master': 4, 'Sir': 12, 'Jonkheer': 16, 'Col': 14, \
        'Major': 10}

        'Lady': 11,
        'Mrs': 2, 'Mme': 8,
        'Ms': 9,
        'Mlle': 13, 'Miss': 3,

        'Capt': 15,
        'Col': 14, 'Major': 10
        'Dr': 7,
        'Rev': 6,
        'Sir': 12,
        'Mr': 1,  'Don': 5,
        'Master': 4,

        'Jonkheer': 16,
        }
        """

        self.df['TitleBin'] = self.df['Title'].dropna()
        self.df['TitleBin'] = self.df['TitleBin'].map(self.unique_titles).astype(int8)
        self.df['TitleBin'].fillna(value=-1)
        self.df.drop('Title', axis=1, inplace=True)

        title_bin_corr = self.df.corr()['TitleBin']
        debug("correlation between TitleBin and Gender = {}".format(title_bin_corr['Gender']))
        debug("correlation between TitleBin and Age = {}".format(title_bin_corr['Age']))

        proposed_correlated_features = {'Gender', 'Age'}
        log_count_feature_by_title_bin = lambda feature: debug(
            "AVERAGE('{}') FROM df GROUP BY 'TitleBin' = {}".format(
                feature, self.df.groupby('TitleBin')[feature].sum()))
        list(map(log_count_feature_by_title_bin, proposed_correlated_features))
        jointplot('Gender', 'TitleBin', self.df)
        tight_layout()
        savefig('_'.join((plot_file_base, 'gender_vs_titlebin')))

        jointplot('Age', 'TitleBin', self.df)
        tight_layout()
        savefig('_'.join((plot_file_base, 'age_vs_titlebin')))

    def enumerate_feature(self, feature, enum_feature):
        self.df[enum_feature] = self.df[feature].dropna()
        self.df[enum_feature] = self.df[enum_feature].map(self.unique_tokens[feature]).astype(int8)
        self.df[enum_feature].fillna(value=-1)
        self.df.drop(feature, axis=1, inplace=True)

    def enumerate_family(self):
        self.enumerate_feature('FamilyName', 'FamilyId')

    def tokenize_title(self):
        """
        TODO suffix titles (e.g. MD, Ph.D., Esq.)
        """

        last_title_enum = 0

        self.df.insert(self.df.ndim, 'Title', None)
        for row in self.df.iterrows():

            series_data = row[1]
            name = series_data['Name']
            if name is None:
                continue

            name_str = str(name)
            if len(name_str) == 0:
                continue

            split_on_commas = name_str.split(sep=", ", maxsplit=1)
            if split_on_commas is None or len(split_on_commas) < 2:
                continue

            after_first_comma = split_on_commas[1]
            if after_first_comma is None:
                continue

            split_on_spaces = after_first_comma.split(sep=' ', maxsplit=1)
            if split_on_spaces is None or len(split_on_spaces) < 2:
                continue

            name_prefix = split_on_spaces[0]
            if len(name_prefix) < 3 or name_prefix[-1] != '.':
                # prefix is not a title (probably first initial)
                continue

            title = name_prefix[:-1].capitalize()
            # series_data['Title'] = title
            self.df.set_value(row[0], 'Title', title)

            if title not in self.unique_titles:
                last_title_enum += 1
                self.unique_titles[title] = last_title_enum

        debug("COUNT(DISTINCT 'Title') FROM df GROUP BY 'Title' = {} (sum={})".format(
            self.df.groupby('Title').Title.count(), self.df.groupby('Title').Title.count().sum()))

    @staticmethod
    def tokenize_title_from_name(name_str):
        split_on_commas = name_str.split(sep=", ", maxsplit=1)
        if split_on_commas is None or len(split_on_commas) < 2:
            return None

        after_first_comma = split_on_commas[1]
        if after_first_comma is None:
            return None

        split_on_spaces = after_first_comma.split(sep=' ', maxsplit=1)
        if split_on_spaces is None or len(split_on_spaces) < 2:
            return None

        name_prefix = split_on_spaces[0]
        if len(name_prefix) < 3 or name_prefix[-1] != '.':
            # prefix is not a title (probably first initial)
            return None

        return name_prefix[:-1].capitalize()

    def tokenize(self, feature, token, tokenization_func):
        last_index = 0
        tokens_dict = self.unique_tokens[token] = {}

        self.df.insert(self.df.ndim, token, None)
        for row in self.df.iterrows():

            series_data = row[1]
            to_tokenize = series_data[feature]
            if to_tokenize is None:
                continue

            str_to_tokenize = str(to_tokenize)
            if len(str_to_tokenize) == 0:
                continue

            extracted_token = tokenization_func(str_to_tokenize)
            if extracted_token is None:
                continue

            self.df.set_value(row[0], token, extracted_token)

            if extracted_token not in tokens_dict:
                last_index += 1
                tokens_dict[extracted_token] = last_index

        debug("COUNT(DISTINCT 'token_name') FROM df GROUP BY 'token_name' = {} (sum={})".format(
            self.df.groupby(token)[token].count(), self.df.groupby(token)[token].count().sum()))

    def tokenize_family(self):

        def family_tokenization(name_str):
            split_on_commas = name_str.split(sep=',', maxsplit=1)
            if split_on_commas is None or len(split_on_commas) < 2:
                return None

            before_first_comma = split_on_commas[0]
            if before_first_comma is None:
                return None

            return before_first_comma.capitalize()

        self.tokenize('Name', 'FamilyName', family_tokenization)


def train(model, X, y):
    model = model.fit(X, y)
    y_hat = model_prediction(model, X, y)
    confusion_matrix(y, y_hat)
    classification_report(y, y_hat)
    return model


def preprocess(csv):
    processor = DataProcessor(csv)
    return processor.transform()


def test(model, ids, X):
    """
    Test, save public test as csv: 'PassengerId', 'Survived'
    """

    y_hat = model_prediction(model, X)

    with(
            open('out/survival_' + str(epoch_seconds_now()) + "_" + type(model).__name__ + '.csv',
                 'w')) as submission_file:
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
        debug("cross-validation scores: {}".format(scores))

        y_hat = cross_val_predict(model, X, y, n_jobs=cpu_count() - 2, verbose=3)
    return y_hat


def model_selection():
    MLDataSet = namedtuple('MLDataSet', ('ids', 'X', 'y'), True)

    ids, X, y = preprocess('data/train.csv')
    train_set = MLDataSet(ids, X, y)

    ids, X, y = preprocess('data/test.csv')
    assert y is None
    test_set = MLDataSet(ids, X, y)

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

    independent_classifiers(classifiers, train_set, test_set)
    voting_classification(classifiers, train_set, test_set)


def independent_classifiers(classifiers, train_set, test_set):
    for model_name in classifiers.keys():
        model = classifiers[model_name]
        info("%s training %s" % (model_name, model))
        evaluate_model(model, train_set, test_set)


def evaluate_model(model, train_set, test_set):
    model = train(model, train_set.X, train_set.y)
    test(model, test_set.ids, test_set.X)


def voting_classification(classifiers, train_set, test_set):
    info("training VotingClassifier")
    voting = VotingClassifier(classifiers.items(), n_jobs=cpu_count() - 2)
    evaluate_model(voting, train_set, test_set)


def setup_logger():
    # TODO current_path =
    log_file_rel_path = sep.join(("out", "log", out_file_no_path + ".log"))

    logger_stream_ = open(log_file_rel_path, "a")
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
