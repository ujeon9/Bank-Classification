# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler
import warnings
warnings.filterwarnings(action='ignore')


def purity(predicted, actual):
    sum = 0
    table = pd.crosstab(predicted, actual)
    table = table.as_matrix()
    for i in range(table.size/2):
        sum = sum + max(table[i][0], table[i][1])

    return float(sum)/float(table.sum())


# load data
data = pd.read_csv('C://work/Country.csv')
print ("----------------------Loaded data----------------------")
print(data)

# drop irrelevant attributes
drop_target = ['CountryCode', 'ShortName', 'TableName', 'LongName', 'Alpha2Code', 'SpecialNotes', 'Wb2Code', 'NationalAccountsBaseYear', 'NationalAccountsReferenceYear',
               'SnaPriceValuation', 'LendingCategory', 'OtherGroups', 'AlternativeConversionFactor', 'PppSurveyYear', 'LatestPopulationCensus',
               'LatestHouseholdSurvey', 'SourceOfMostRecentIncomeAndExpenditureData', 'VitalRegistrationComplete',
               'LatestAgriculturalCensus', 'LatestIndustrialData', 'LatestTradeData', 'LatestWaterWithdrawalData']
data = data.drop(drop_target, axis=1)
print ("----------------------Column dropped data----------------------")
print(data)

# Cleansing data
data.dropna(how='all', axis=0)
data.fillna(method='ffill', inplace=True)

print ("----------------------Cleansed data----------------------")
print(data)

# Label encode data
data = data.apply(LabelEncoder().fit_transform)
print ("----------------------Encoded data----------------------")
print(data)


# race_for_out = data
# # Isolation Forest 방법을 사용하기 위해, 변수로 선언을 해 준다.
# clf = IsolationForest(max_samples=1000, random_state=1)
#
# # fit 함수를 이용하여, 데이터셋을 학습시킨다. race_for_out은 dataframe의 이름이다.
# clf.fit(race_for_out)
#
# # predict 함수를 이용하여, outlier를 판별해 준다. 0과 1로 이루어진 Series형태의 데이터가 나온다.
# y_pred_outliers = clf.predict(race_for_out)
#
#
# # 원래의 dataframe에 붙이기. 데이터가 0인 것이 outlier이기 때문에, 0인 것을 제거하면 outlier가 제거된  dataframe을 얻을 수 있다.
# out = pd.DataFrame(y_pred_outliers)
# out = out.rename(columns={0: "out"})
# data = pd.concat([race_for_out, out], 1)


# # Label encode data
# encoder = OneHotEncoder().fit(data)
# data = encoder.transform(data)
# print("***Encoded data***")
# print(data)


# split data
y = data["SystemOfTrade"]
X = data.drop('SystemOfTrade', axis=1)


# # Min-Max scale data
# X = MinMaxScaler().fit(X).transform(X)
# print ("----------------------Scaled data----------------------")
# print(X)

# # Standard scale data
# X = StandardScaler().fit(X).transform(X)
# print ("----------------------Scaled data----------------------")
# print(X)

# # Robust scale data
# X = RobustScaler().fit(X).transform(X)
# print ("----------------------Scaled data----------------------")
# print(X)


# Reduce dimensionality & visualizable
X = PCA(n_components=2).fit_transform(X)
X = pd.DataFrame(X)
X.columns = ['P1', 'P2']


# DBSCAN
eps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [3, 5, 10, 15, 20, 30, 50, 100]
metrics = ["euclidean", "hamming"]


db_default = DBSCAN(eps=0.001, min_samples=100, metric="euclidean").fit(X)
labels = db_default.labels_
print("eps={} min_samples={} metric={} purity = {}".format(EPS, SAMPLES, METRIC, purity(labels, y)))

# Plot result
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# # y_pred = np.hstack([DBSCAN(eps=0.2, min_samples=3, metric="euclidean").fit_predict(X).reshape(-1, 1)])
# db = DBSCAN(eps=0.2, min_samples=3, metric="euclidean")
# db.fit(X)
# print(db.labels_)
# y_pred = db.fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired')
# plt.title('DBSCAN')
# plt.savefig('DBSCAN.png')

# visualization
df = np.hstack([X, DBSCAN(eps=0.2, min_samples=3, metric="euclidean").fit_predict(X).reshape(-1, 1)])
df_ft0 = df[df[:, 2]==0, :] # 클러스터 0 추출
df_ft1 = df[df[:, 2]==1, :] # 클러스터 1 추출
df_ft2 = df[df[:, 2]==2, :] # 클러스터 2 추출
df_ft3 = df[df[:, 2]==3, :] # 클러스터 2 추출
df_ft4 = df[df[:, 2]==4, :] # 클러스터 2 추출
df_ft5 = df[df[:, 2]==5, :] # 클러스터 2 추출
df_ft6 = df[df[:, 2]==6, :] # 클러스터 2 추출
df_ft7 = df[df[:, 2]==7, :] # 클러스터 2 추출
df_ft8 = df[df[:, 2]==8, :] # 클러스터 2 추출
df_ft9 = df[df[:, 2]==9, :] # 클러스터 2 추출
df_ft10 = df[df[:, 2]==10, :] # 클러스터 2 추출
df_ft11 = df[df[:, 2]==11, :] # 클러스터 2 추출
df_ft12 = df[df[:, 2]==12, :] # 클러스터 2 추출
df_ft13 = df[df[:, 2]==13, :] # 클러스터 2 추출

# matplotlib로 그래프 그리기
plt.scatter(df_ft0[:, 0], df_ft0[:, 1], label='cluster 0', cmap='Pairs')
plt.scatter(df_ft1[:, 0], df_ft1[:, 1], label='cluster 1', cmap='Pairs')
plt.scatter(df_ft2[:, 0], df_ft2[:, 1], label='cluster 2', cmap='Pairs')
plt.scatter(df_ft3[:, 0], df_ft3[:, 1], label='cluster 3', cmap='Pairs')
plt.scatter(df_ft4[:, 0], df_ft4[:, 1], label='cluster 4', cmap='Pairs')
plt.scatter(df_ft5[:, 0], df_ft5[:, 1], label='cluster 5', cmap='Pairs')
plt.scatter(df_ft6[:, 0], df_ft6[:, 1], label='cluster 6', cmap='Pairs')
plt.scatter(df_ft7[:, 0], df_ft7[:, 1], label='cluster 7', cmap='Pairs')
plt.scatter(df_ft8[:, 0], df_ft8[:, 1], label='cluster 8', cmap='Pairs')
plt.scatter(df_ft9[:, 0], df_ft9[:, 1], label='cluster 9', cmap='Pairs')
plt.scatter(df_ft10[:, 0], df_ft10[:, 1], label='cluster 10', cmap='Pairs')
plt.scatter(df_ft11[:, 0], df_ft11[:, 1], label='cluster 11', cmap='Pairs')
plt.scatter(df_ft12[:, 0], df_ft12[:, 1], label='cluster 12', cmap='Pairs')
plt.scatter(df_ft13[:, 0], df_ft13[:, 1], label='cluster 13', cmap='Pairs')
plt.xlabel('feature 0')
plt.ylabel('feature 1')
# plt.legend()
plt.savefig('DBSCAN.png')
