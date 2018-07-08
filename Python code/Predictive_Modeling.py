from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
from Generic_Functions import model_fitting_and_get_training_accuracy, get_test_accuracy, clean_data, outlier_removal

or_test = pd.read_csv('C:/Users/Dell/Downloads/Test_data.csv')
or_train = pd.read_csv('C:/Users/Dell/Downloads/Train_data.csv')

cat = ['state', 'area code', 'phone number', 'international plan', 'voice mail plan']
num = list(set(or_train.columns) - set(cat+['Churn']))

train = clean_data(or_train)
test = clean_data(or_test)

# Remove three discrete numerical variables
# train = train.drop(['total intl calls', 'number vmail messages', 'number customer service calls'], axis=1)
# test = test.drop(['total intl calls', 'number vmail messages', 'number customer service calls'], axis=1)

# Dataset without any outliers
# train = outlier_removal(train, num)

x_train = train.iloc[:, train.columns != 'Churn']
y_train = train.iloc[:, train.columns == 'Churn']
x_test = test.iloc[:, test.columns != 'Churn']
y_test = test.iloc[:, test.columns == 'Churn']

# --------------------------------- Normalizing data with mean 0 and variance 1 ----------------------------------------

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# --------------------------------------------- Logistic Regression ----------------------------------------------------

lr_predictions_train, lr_predictions_train_prob, lr_accuracy_train, lr_classifier, lr_average_precision_train = \
    model_fitting_and_get_training_accuracy(LogisticRegression, x_train, y_train, random_state=0)

lr_predictions_test, lr_predictions_test_prob, lr_accuracy_test, lr_average_precision_test, lr_precision, lr_recall = \
    get_test_accuracy(x_test, lr_classifier, y_test)

# ------------------------------------------------- Naive Bayes --------------------------------------------------------

gnb_predictions_train, gnb_predictions_train_prob, gnb_accuracy_train, gnb_classifier, gnb_average_precision_train = \
    model_fitting_and_get_training_accuracy(GaussianNB, x_train, y_train)

gnb_predictions_test, gnb_predictions_test_prob, gnb_accuracy_test, gnb_average_precision_test, gnb_precision, \
    gnb_recall = get_test_accuracy(x_test, gnb_classifier, y_test)

# -------------------------------------------- Support Vector Machine --------------------------------------------------

svm_predictions_train, svm_predictions_train_prob, svm_accuracy_train, svm_classifier, svm_average_precision_train = \
    model_fitting_and_get_training_accuracy(SVC, x_train, y_train, probability=True, random_state=0)

svm_predictions_test, svm_predictions_test_prob, svm_accuracy_test, svm_average_precision_test, svm_precision, \
    svm_recall = get_test_accuracy(x_test, svm_classifier, y_test)

# -------------------------------------------- Gradient Boosted Tree ---------------------------------------------------

gbc_predictions_train, gbc_predictions_train_prob, gbc_accuracy_train, gbc_classifier, gbc_average_precision_train = \
    model_fitting_and_get_training_accuracy(GradientBoostingClassifier, x_train, y_train, max_depth=2, random_state=0)

gbc_predictions_test, gbc_predictions_test_prob, gbc_accuracy_test, gbc_average_precision_test, gbc_precision, \
    gbc_recall = get_test_accuracy(x_test, gbc_classifier, y_test)

# ------------------------------------------------ Decision Tree -------------------------------------------------------

dtc_predictions_train, dtc_predictions_train_prob, dtc_accuracy_train, dtc_classifier, dtc_average_precision_train = \
    model_fitting_and_get_training_accuracy(DecisionTreeClassifier, x_train, y_train, min_impurity_decrease=0.01,
                                            criterion='entropy', random_state=0)

dtc_predictions_test, dtc_predictions_test_prob, dtc_accuracy_test, dtc_average_precision_test, dtc_precision, \
    dtc_recall = get_test_accuracy(x_test, dtc_classifier, y_test)

# ------------------------------------------------ Random Forest -------------------------------------------------------

rf_predictions_train, rf_predictions_train_prob, rf_accuracy_train, rf_classifier, rf_average_precision_train = \
    model_fitting_and_get_training_accuracy(RandomForestClassifier, x_train, y_train, n_estimators=500, random_state=0)

rf_predictions_test, rf_predictions_test_prob, rf_accuracy_test, rf_average_precision_test, rf_precision, rf_recall = \
    get_test_accuracy(x_test, rf_classifier, y_test)

# ------------------------------------ Precision Recall Curves of each model -------------------------------------------

plt.plot(lr_recall, lr_precision, color='b', label='Logistic Regression')
plt.plot(gnb_recall, gnb_precision, color='k', label='Naive Bayes')
plt.plot(svm_recall, svm_precision, color='g', label='Support Vector Machine')
plt.plot(gbc_recall, gbc_precision, color='m', label='Gradient Boosted Tree')
plt.plot(dtc_recall, dtc_precision, color='r', label='Decision Tree')
plt.plot(rf_recall, rf_precision, color='y', label='Random Forest')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1])
plt.title('Precision-Recall curves for every model')
plt.legend(loc='bottom left')
plt.show()

# ----------------------------------- Write the predictions in a csv file with features used ---------------------------

test['rf_predictions_test'] = rf_predictions_test
test.to_csv('C:/Users/Dell/Desktop/Python_Churn_test_set_with_predictions.csv', index=False)

# ----------------------------------------- Print performance metrics of the best model --------------------------------

print('Training set accuracy of the best model i.e. Random Forest is : ', rf_accuracy_train)
print('Test set accuracy of the best model i.e. Random Forest is : ', rf_accuracy_test)
print('Average precision score of the best model i.e. Random Forest on training set is : ', rf_average_precision_train)
print('Average precision score of the best model i.e. Random Forest on test set is : ', rf_average_precision_test)

# ----------------------------------------------------- End ------------------------------------------------------------
