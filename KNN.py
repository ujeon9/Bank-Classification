#Pandas for dataframes
import pandas as pd
#Changing default display option to display all columns
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_columns', 21)

#Numpy for numerical computing
import numpy as np


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
DF = df.copy()

DF = DF[DF.job != 'unknown']
DF = DF[DF.marital != 'unknown']
DF = DF[DF.education != 'unknown']
DF = DF[DF.education != 'illiterate']
del DF['default']
del DF['duration']
DF = DF[DF.housing != 'unknown']
DF = DF[DF.loan != 'unknown']

DF.job.replace(['entrepreneur', 'self-employed'], 'self-employed', inplace=True)
DF.job.replace(['admin.', 'management'], 'administration_management', inplace=True)
DF.job.replace(['blue-collar', 'technician'], 'blue-collar', inplace=True)
DF.job.replace(['retired', 'unemployed'], 'no_active_income', inplace=True)
DF.job.replace(['services', 'housemaid'], 'services', inplace=True)
DF.education.replace(['basic.9y', 'basic.6y', 'basic.4y'], 'basic_school', inplace=True)

#Getting the positions of the mistakenly labeled 'pdays'
ind_999 = DF.loc[(DF['pdays'] == 999) & (df['poutcome'] != 'nonexistent')]['pdays'].index.values
#Assigning NaNs instead of '999'
DF.loc[ind_999, 'pdays'] = np.nan

#Dropping NAs from the dataset
DF = DF.dropna()
#Saving the cleaned dataset as a file
DF.to_csv('C://work/cleaned_data.csv')
#Substituting the string predictor variable values with numbers
DF.y.replace(['yes'], 1, inplace=True)
DF.y.replace(['no'], 0, inplace=True)

encoder = LabelEncoder()
y=DF['marital'].values
y=encoder.fit_transform(y)
df_cleaned1 = pd.get_dummies(DF, drop_first=True)
# Separate target data

X=df_cleaned1
for i in range (0,19):
    encoder = LabelEncoder()
    if(isinstance(X.iloc[0,i],str)):
        tmp=encoder.fit_transform(X.iloc[:,i])
        X.iloc[:,i]=tmp

scaler=MinMaxScaler()
X=scaler.fit_transform(X)

# Split datsaset into train and test data ( for Holdout Evaluation )
X_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=1)
# Create a KNN classifier
knn=KNeighborsClassifier(n_neighbors=10)
# Train the KNN classifier
knn.fit(X_train,y_train)

# show the first 5 model predictions on the test data
print(knn.predict(x_test[0:10]))
# Check accuracy of the model on the test data
print(knn.score(x_test,y_test))

# Create a new KNN model ( for k-fold cross validation )
knn_cv = KNeighborsClassifier(n_neighbors=10)
#  Train model with cv of 5
cv_scores = cross_val_score(knn_cv, x_test, y_test, cv=5)
# print each cv accuracy and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

