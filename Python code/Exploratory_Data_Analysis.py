import pandas as pd
import numpy as np
from scipy.stats import norm, chi2_contingency
import matplotlib.pyplot as plt

train = pd.read_csv('C:/Users/Dell/Downloads/Train_data.csv')

target_distribution = train['Churn'].value_counts()

# Unique value counts of target variable
print(target_distribution)

cat = ['state', 'area code', 'phone number', 'international plan', 'voice mail plan']

# ------------------------------------- Chi-square test for every categorical variable ---------------------------------

for i in cat:
    print(i)
    cross_tab = pd.crosstab(train[i], train['Churn'])
    chi2, p, dof, ex = chi2_contingency(cross_tab)
    print(p)

# -------------- Histogram of every numerical variable with Normal distribution curve superimposed over it -------------

num = list(set(train.columns) - set(cat+['Churn']))

fig = plt.figure()
for i, var_name in enumerate(num):
    mu, std = norm.fit(train[var_name])
    ax = fig.add_subplot(5, 3, i+1)
    train[var_name].hist(density=True, edgecolor='k', color='w', ax=ax)
    xmin, xmax = min(train[var_name]), max(train[var_name])
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_title(var_name)
fig.tight_layout()  # Reduces size of each plot to fit each variable name thereby improving the appearance
plt.show()

# ----------------------------------------- Correlation plot of numerical variables ------------------------------------

plt.matshow(train[num].corr())
plt.xticks(range(len(train[num].columns)), train[num].columns, fontsize=8, rotation=90)
plt.yticks(range(len(train[num].columns)), train[num].columns)
plt.colorbar()
plt.show()

# ------------------------------------------ Box plot of every numerical variable --------------------------------------

fig = plt.figure()
for i, var_name in enumerate(num):
    ax = fig.add_subplot(5, 3, i+1)
    train.boxplot(column=var_name, ax=ax, flierprops=dict(marker='.', markerfacecolor='black', markersize=4))
fig.tight_layout()  # Reduces size of each plot to fit each variable name thereby improving the appearance
plt.show()

# ----------------------------------------------------- End ------------------------------------------------------------
