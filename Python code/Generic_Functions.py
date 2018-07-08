from sklearn.metrics import average_precision_score, accuracy_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import numpy as np


def model_fitting_and_get_training_accuracy(model, x_train, y_train, **kwargs):

    classifier = model(**kwargs)

    classifier.fit(x_train, y_train)

    predictions_train = classifier.predict(x_train)

    predictions_train_prob = classifier.predict_proba(x_train)[:, 1]

    accuracy_train = accuracy_score(predictions_train, y_train)

    average_precision_train = average_precision_score(y_train, predictions_train_prob)

    return predictions_train, predictions_train_prob, round(accuracy_train, 3), classifier, \
        round(average_precision_train, 3)


def get_test_accuracy(x_test, classifier, y_test):

    predictions_test = classifier.predict(x_test)

    predictions_test_prob = classifier.predict_proba(x_test)[:, 1]

    accuracy_test = accuracy_score(predictions_test, y_test)

    average_precision_test = average_precision_score(y_test, predictions_test_prob)

    precision, recall, _ = precision_recall_curve(y_test, predictions_test_prob)

    return predictions_test, predictions_test_prob, round(accuracy_test, 3), round(average_precision_test, 3), \
        precision, recall


def outlier_removal(data, cols):
    for i in cols:
        print(i)
        q75, q25 = np.percentile(data.loc[:, i], [75, 25])
        iqr = q75-q25
        min_value = q25 - (1.5*iqr)
        max_value = q75 + (1.5*iqr)
        print(q75, q25, iqr, min_value, max_value)
        data = data.drop(data[data.loc[:, i] < min_value].index)
        data = data.drop(data[data.loc[:, i] > max_value].index)
        print(data.shape[0])
    return data


def clean_data(data, remove=None):
    cleansed_data = data.copy()
    cleansed_data = cleansed_data.drop(['state', 'phone number', 'area code'], axis=1)
    if remove == 'charges':
        cleansed_data = cleansed_data.drop(['total night charge', 'total day charge', 'total intl charge',
                                            'total eve charge'], axis=1)
    if remove == 'minutes':
        cleansed_data = cleansed_data.drop(['total night minutes', 'total day minutes', 'total intl minutes',
                                            'total eve minutes'], axis=1)
    le = LabelEncoder()
    cleansed_data['Churn'] = le.fit_transform(cleansed_data['Churn'])
    cleansed_data['international plan'] = le.fit_transform(cleansed_data['international plan'])
    cleansed_data['voice mail plan'] = le.fit_transform(cleansed_data['voice mail plan'])
    return cleansed_data

# ----------------------------------------------------- End ------------------------------------------------------------
