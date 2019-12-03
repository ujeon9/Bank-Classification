#Pandas for dataframes
import pandas as pd
#Changing default display option to display all columns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

pd.set_option('display.max_columns', 21)

#Numpy for numerical computing
import numpy as np
#Matplotlib for visualization
import matplotlib.pyplot as plt

#Seaborn for easier visualization
import seaborn as sn

#Machine learning packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
#Loading the data set
df = pd.read_csv('C://work/bank-additional-full.csv', sep=';')
#Dropping the duplicates
df = df.drop_duplicates()
#Selecting categorical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
#Looping through the columns and changing type to 'category'
for column in categorical_columns:
    df[column] = df[column].astype('category')

#Creating a copy of the original data frame
df_cleaned = df.copy()
#Dropping the unknown job level
df_cleaned = df_cleaned[df_cleaned.job != 'unknown']
#Dropping the unknown marital status
df_cleaned = df_cleaned[df_cleaned.marital != 'unknown']
#Dropping the unknown and illiterate education level
df_cleaned = df_cleaned[df_cleaned.education != 'unknown']
df_cleaned = df_cleaned[df_cleaned.education != 'illiterate']
#Deleting the 'default' column
del df_cleaned['default']
#Deleting the 'duration' column
del df_cleaned['duration']
#Dropping the unknown housing loan status
df_cleaned = df_cleaned[df_cleaned.housing != 'unknown']
#Dropping the unknown personal loan status
df_cleaned = df_cleaned[df_cleaned.loan != 'unknown']
#Combining entrepreneurs and self-employed into self-employed
df_cleaned.job.replace(['entrepreneur', 'self-employed'], 'self-employed', inplace=True)
#Combining administrative and management jobs into admin_management
df_cleaned.job.replace(['admin.', 'management'], 'administration_management', inplace=True)
#Combining blue-collar and tecnician jobs into blue-collar
df_cleaned.job.replace(['blue-collar', 'technician'], 'blue-collar', inplace=True)
#Combining retired and unemployed into no_active_income
df_cleaned.job.replace(['retired', 'unemployed'], 'no_active_income', inplace=True)
#Combining services and housemaid into services
df_cleaned.job.replace(['services', 'housemaid'], 'services', inplace=True)

#Combining basic school degrees
df_cleaned.education.replace(['basic.9y', 'basic.6y', 'basic.4y'], 'basic_school', inplace=True)

#Getting the positions of the mistakenly labeled 'pdays'
ind_999 = df_cleaned.loc[(df_cleaned['pdays'] == 999) & (df['poutcome'] != 'nonexistent')]['pdays'].index.values
#Assigning NaNs instead of '999'
df_cleaned.loc[ind_999, 'pdays'] = np.nan

#Dropping NAs from the dataset
df_cleaned = df_cleaned.dropna()
#Saving the cleaned dataset as a file
df_cleaned.to_csv('C://work/cleaned_data.csv')
#Substituting the string predictor variable values with numbers
df_cleaned.y.replace(['yes'], 1, inplace=True)
df_cleaned.y.replace(['no'], 0, inplace=True)

encoder = LabelEncoder()
y=df_cleaned['marital'].values
y=encoder.fit_transform(y)
df_cleaned1 = pd.get_dummies(df_cleaned, drop_first=True)
# Separate target data

X=df_cleaned1


# Split datsaset into train and test data ( for Holdout Evaluation )
X_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=5)
# Create a LogisticRegression classifier
logmodel = LogisticRegression(C=0.001, solver='liblinear', max_iter=50)
#liblinear은 더 낮음
logmodel.fit(X_train, y_train)


#  Train model with cv of 5
cv_scores = cross_val_score(logmodel, x_test, y_test, cv=5)
# print each cv accuracy and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

