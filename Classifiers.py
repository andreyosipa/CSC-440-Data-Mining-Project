
# coding: utf-8

# In[463]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, KFold


# In[527]:

import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams.update({'font.size': 20})


# In[2]:

get_ipython().magic(u'matplotlib inline')


# ##Divide data to test and train

# In[3]:

data = pd.read_csv("train.csv")
data_array = np.array(data)[:,1:]


# In[4]:

num = len(data)


# In[5]:

np.random.shuffle(data_array)
train_array = data_array[:np.round(0.8 * num),:]
test_array = data_array[np.round(0.8 * num):,:]


# In[6]:

plt.figure(figsize=(25,10))
plt.plot(np.sort(train_array[[np.random.choice(train_array.shape[0], test_array.shape[0])],-1][0]), label='train')
plt.plot(np.sort(test_array[:,-1]), label='train')
plt.legend()


# test that both have same distributions

# ##Perform simple str -> int encoding of all data in arrays.

# In[7]:

columns = list(data.columns)[1:]


# In[8]:

fixed_range_attributes = {
    u'MSSubClass': [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
    u'MSZoning': ['A', 'C (all)', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
    u'Street' : [u'Grvl', u'Pave'],
    u'Alley' : [u'Grvl', u'Pave', u'NA'],
    u'LotShape' : [u'Reg', u'IR1', u'IR2', u'IR3'],
    u'LandContour' : [u'Lvl', u'Bnk', u'HLS', u'Low'],
    u'Utilities' : [u'AllPub', u'NoSewr', u'NoSeWa', u'ELO'],
    u'LotConfig' : [u'Inside', u'Corner', u'CulDSac', u'FR2', u'FR3'],
    u'LandSlope' : [u'Gtl', u'Mod', u'Sev'],
    u'Neighborhood' : [u'Blmngtn', u'Blueste', u'BrDale', u'BrkSide', u'ClearCr', u'CollgCr', u'Crawfor', u'Edwards', 
                       u'Gilbert', u'Greens', u'GrnHill', u'IDOTRR', u'Landmrk', u'MeadowV',u'Mitchel', u'NAmes', 
                       u'NoRidge', u'NPkVill', u'NridgHt', u'NWAmes', u'OldTown', u'SWISU', u'Sawyer', u'SawyerW', 
                       u'Somerst', u'StoneBr', u'Timber', u'Veenker'],
    u'Condition1' : [u'Artery', u'Feedr', u'Norm', u'RRNn', u'RRAn', u'PosN', u'PosA', u'RRNe', u'RRAe'],
    u'Condition2' : [u'Artery', u'Feedr', u'Norm', u'RRNn', u'RRAn', u'PosN', u'PosA', u'RRNe', u'RRAe'],
    u'BldgType' : [u'1Fam', u'2fmCon', u'Duplex', u'TwnhsE', u'Twnhs'],
    u'HouseStyle' : [u'1Story', u'1.5Fin', u'1.5Unf', u'2Story', u'2.5Fin', u'2.5Unf', u'SFoyer', u'SLvl'],
    u'RoofStyle' : [u'Flat', u'Gable', u'Gambrel', u'Hip', u'Mansard', u'Shed'],
    u'RoofMatl' : [u'ClyTile', u'CompShg', u'Membran', u'Metal', u'Roll', u'Tar&Grv', u'WdShake', u'WdShngl'],
    u'Exterior1st' : [u'AsbShng', u'AsphShn', u'BrkComm', u'BrkFace', u'CBlock', u'CemntBd', u'HdBoard', u'ImStucc',
                u'MetalSd', u'Other', u'Plywood', u'PreCast', u'Stone', u'Stucco', u'VinylSd', u'Wd Sdng', u'WdShing'],
    u'Exterior2nd' : [u'AsbShng', u'AsphShn', u'Brk Cmn', u'BrkFace', u'CBlock', u'CmentBd', u'HdBoard', u'ImStucc',
                u'MetalSd', u'Other', u'Plywood', u'PreCast', u'Stone', u'Stucco', u'VinylSd', u'Wd Sdng', u'Wd Shng'],
    u'MasVnrType' : [u'BrkCmn', u'BrkFace', u'CBlock', u'None', u'Stone'],
    u'ExterQual' : [u'Ex', u'Gd', u'TA', u'Fa', u'Po'],
    u'ExterCond' : [u'Ex', u'Gd', u'TA', u'Fa', u'Po'],
    u'Foundation' : [u'BrkTil', u'CBlock', u'PConc', u'Slab', u'Stone', u'Wood'],
    u'BsmtQual' : [u'Ex', u'Gd', u'TA', u'Fa', u'Po', u'NA'],
    u'BsmtCond' : [u'Ex', u'Gd', u'TA', u'Fa', u'Po', u'NA'],
    u'BsmtExposure' : [u'Gd', u'Av', u'Mn', u'No', u'NA'],
    u'BsmtFinType1' : [u'GLQ', u'ALQ', u'BLQ', u'Rec', u'LwQ', u'Unf', u'NA'],
    u'BsmtFinType2' : [u'GLQ', u'ALQ', u'BLQ', u'Rec', u'LwQ', u'Unf', u'NA'],
    u'Heating' : [u'Floor', u'GasA', u'GasW', u'Grav', u'OthW', u'Wall'],
    u'HeatingQC' : [u'Ex', u'Gd', u'TA', u'Fa', u'Po'],
    u'CentralAir' : [u'N', u'Y'],
    u'Electrical' : [u'SBrkr', u'FuseA', u'FuseF', u'FuseP', u'Mix'],
    u'KitchenQual' : [u'Ex', u'Gd', u'TA', u'Fa', u'Po'],
    u'Functional' : [u'Typ', u'Min1', u'Min2', u'Mod', u'Maj1', u'Maj2', u'Sev', u'Sal'],
    u'FireplaceQu' : [u'Ex', u'Gd', u'TA', u'Fa', u'Po', u'NA'],
    u'GarageType' : [u'2Types', u'Attchd', u'Basment', u'BuiltIn', u'CarPort', u'Detchd', u'NA'],
    u'GarageFinish' : [u'Fin', u'RFn', u'Unf', u'NA'],
    u'GarageQual' : [u'Ex', u'Gd', u'TA', u'Fa', u'Po', u'NA'],
    u'GarageCond' : [u'Ex', u'Gd', u'TA', u'Fa', u'Po', u'NA'],
    u'PavedDrive' : [u'N', u'Y', u'P'],
    u'PoolQC' : [u'Ex', u'Gd', u'TA', u'Fa', u'NA'],
    u'Fence' : [u'GdPrv', u'MnPrv', u'GdWo', u'MnWw', u'NA'],
    u'MiscFeature' : [u'Elev', u'Gar2', u'Othr', u'Shed', u'TenC', u'NA'],
    u'SaleType' : [u'WD', u'CWD', u'VWD', u'New', u'COD', u'Con', u'ConLw', u'ConLI', u'ConLD', u'Oth'],
    u'SaleCondition' :[u'Normal', u'Abnorml', u'AdjLand', u'Alloca', u'Family', u'Partial']
}


# In[9]:

#attributes by categories
continuous_attributes = [
u'LotFrontage',
u'LotArea',
u'MasVnrArea',
u'BsmtFinSF1',
u'BsmtFinSF2',
u'BsmtUnfSF', 
u'TotalBsmtSF', 
u'1stFlrSF',
u'2ndFlrSF',
u'LowQualFinSF',
u'GrLivArea',
u'GarageArea',
u'WoodDeckSF',
u'OpenPorchSF',
u'EnclosedPorch',
u'3SsnPorch',
u'ScreenPorch',
u'PoolArea',
u'MiscVal',
u'SalePrice']

ordinal_attributes = [
u'LotShape',
u'Utilities',
u'LandSlope',
u'OverallQual',
u'OverallCond',
u'ExterQual',
u'ExterCond',
u'BsmtQual',
u'BsmtCond',
u'BsmtExposure',
u'BsmtFinType1',
u'BsmtFinType2',
u'HeatingQC',
u'Electrical',
u'KitchenQual',
u'Functional',
u'FireplaceQu',
u'GarageFinish',
u'GarageQual',
u'GarageCond',
u'PavedDrive',
u'PoolQC',
u'Fence']

nominal_attributes = [
u'MSSubClass',
u'MSZoning',
u'Street',
u'Alley',
u'LandContour',
u'LotConfig',
u'Neighborhood',
u'Condition1',
u'Condition2',
u'BldgType',
u'HouseStyle',
u'RoofStyle',
u'RoofMatl',
u'Exterior1st',
u'Exterior2nd',
u'MasVnrType',
u'Foundation',
u'Heating',
u'CentralAir',
u'GarageType',
u'MiscFeature',
u'SaleType',
u'SaleCondition']

discrete_attributes = [
u'YearBuilt',
u'YearRemodAdd',
u'BsmtFullBath',
u'BsmtHalfBath',
u'FullBath',
u'HalfBath',
u'BedroomAbvGr',
u'KitchenAbvGr',
u'TotRmsAbvGrd',
u'Fireplaces',
u'GarageYrBlt',
u'GarageCars',
u'MoSold',
u'YrSold']

char_attributes = [
u'MSZoning',
u'Street',
u'Alley',
u'LotShape',
u'LandContour',
u'Utilities',
u'LotConfig',
u'LandSlope',
u'Neighborhood',
u'Condition1',
u'Condition2',
u'BldgType', 
u'HouseStyle',
u'RoofStyle',
u'RoofMatl',
u'Exterior1st',
u'Exterior2nd',
u'MasVnrType',
u'ExterQual',
u'ExterCond',
u'Foundation',
u'BsmtQual',
u'BsmtCond',
u'BsmtExposure',
u'BsmtFinType1',
u'BsmtFinType2',
u'Heating',
u'HeatingQC',
u'CentralAir',
u'Electrical',
u'KitchenQual',
u'Functional',
u'FireplaceQu',
u'GarageType',
u'GarageFinish',
u'GarageQual',
u'GarageCond',
u'PavedDrive',
u'PoolQC',
u'Fence',
u'MiscFeature',
u'SaleType',
u'SaleCondition']


# Label encoders.

# In[10]:

les = {}
for attr in fixed_range_attributes.keys():
    les[attr] = preprocessing.LabelEncoder()
    les[attr].fit(fixed_range_attributes[attr] + ['MISSING'])


# Preprocess data, replace NaNs.

# In[11]:

def preprocess_array(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if columns[j] in char_attributes and type(array[i,j])!=str:
                array[i,j] = 'MISSING'
            else:
                if not columns[j] in char_attributes and np.isnan(array[i,j]):
                    array[i,j] = 0
    return array


# In[12]:

def convert_labels(array):
    converted_array = np.zeros(array.shape)
    for attr_num in range(array.shape[1]): 
        if columns[attr_num] in les.keys():
            converted_array[:,attr_num] = les[columns[attr_num]].transform([str(x) for x in array[:,attr_num]])
        else:
            converted_array[:,attr_num] = array[:,attr_num]
    return converted_array


# In[13]:

train_array = preprocess_array(train_array)
test_array = preprocess_array(test_array)
converted_train_array = convert_labels(train_array)
converted_test_array = convert_labels(test_array)


# ####Note: this encoding ignores non-ordered features as it gives them some ordering.

# ##Evaluator
# Root mean squared error between logarithms of real and predicted values.

# In[14]:

def eval_rmse(prediction, ground_truth):
    return np.sqrt(np.sum(np.power(np.log(prediction) - np.log(ground_truth),2))/len(prediction))


# In[15]:

#function to save prediction result in Kaggle needed format
def save_submission(fname,indexes,prediction):
    f = open(fname,'w')
    f.write('Id,SalePrice\n')
    for idx in range(len(prediction)):
        f.write(str(indexes[idx]) + ',' + str(prediction[idx])+'\n')
    f.close()


# #Test Data

# In[22]:

test_data = pd.read_csv("test.csv")
test_fin_array = np.array(test_data)[:,1:]
test_fin_array = preprocess_array(test_fin_array)
test_fin_array_converted = convert_labels(test_fin_array)
indexes = np.array(test_data)[:,0]


# In[193]:

#Function ro plot truth and prediction. Truth is sorted from lowest to highest.
def price_comparation_graph(truth, model, model_name=None):
    order = np.argsort(truth)
    truth = [truth[i] for i in order]
    model = [model[i] for i in order]
    plt.figure(figsize=(25,15))
    plt.plot(truth, 'ro-')
    plt.plot(model, 'go-', label=model_name)
    plt.legend()


# # DRAFT: PROVIDING ONEHOT ENCODING FOR FEATURES

# In[470]:

columns_active = list(np.array(columns)[kbest_result_idx])
onehot_candidates_idx = [columns_active.index(attr) for attr in nominal_attributes if attr in columns_active]
print onehot_candidates_idx
onehot_encoder = OneHotEncoder(categorical_features=onehot_candidates_idx, sparse=False, handle_unknown='ignore')
onehot_encoder.fit(converted_train_array[:,kbest_result_idx])
train_array_onehot = onehot_encoder.transform(converted_train_array[:,kbest_result_idx+[-1]])
test_array_onehot = onehot_encoder.transform(converted_test_array[:,kbest_result_idx+[-1]])
test_fin_array_onehot = onehot_encoder.transform(test_fin_array_converted[:,kbest_result_idx])


# # Classifiers

# ##Random forest classifier

# In[520]:

random_forest = RandomForestClassifier(n_estimators=35, criterion='gini', max_depth=None, min_samples_split=2, 
                                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                       max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
                                       random_state=None, verbose=0, warm_start=False, class_weight=None)


# In[521]:

random_forest.fit(converted_train_array[:,:-1], converted_train_array[:,-1])


# In[522]:

eval_rmse(random_forest.predict(converted_test_array[:,:-1]),converted_test_array[:,-1])


# In[525]:

save_submission("rf_fin.csv",indexes,random_forest.predict(test_fin_array_converted))


# In[528]:

price_comparation_graph(random_forest.predict(converted_test_array[:,:-1]), converted_test_array[:,-1])


# ##Elastic Net Regression Classifier

# In[530]:

from sklearn.linear_model import ElasticNet


# In[531]:

enet = ElasticNet(alpha=0.9, l1_ratio=0.7)
enet.fit(converted_train_array[:,:-1], converted_train_array[:,-1])


# In[532]:

eval_rmse(enet.predict(converted_test_array[:,:-1]),converted_test_array[:,-1])


# In[533]:

price_comparation_graph(enet.predict(converted_test_array[:,:-1]), converted_test_array[:,-1])


# ###RF submission

# In[26]:

prediction = random_forest.predict(test_fin_array_converted)


# In[27]:

save_submission('submission0RF.csv',indexes,prediction)


# This submission got RMSE = 0.36334 and was ranked 1882. :)

# ###Elastic Net Regression Submission

# In[28]:

prediction = enet.predict(test_fin_array_converted)


# In[29]:

save_submission('submission1ENET.csv',indexes,prediction)


# This submission got RMSE=0.16544 and was ranked 1513.

# In[30]:

from sklearn import grid_search


# In[31]:

parameters = {'alpha' : [0,0.1,0.5,0.9,1,2,5], 
              'l1_ratio' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
             'normalize' : [True, False],
             'selection' : ['random', 'cyclic']}


# In[32]:

enet_grid = ElasticNet()
clf = grid_search.GridSearchCV(enet_grid, parameters)
clf.fit(converted_train_array[:,:-1], converted_train_array[:,-1])


# In[33]:

clf.get_params()


# In[34]:

enet_grid = ElasticNet(alpha=1, l1_ratio=0.5)
enet_grid.fit(converted_train_array[:,:-1], converted_train_array[:,-1])


# In[35]:

print eval_rmse(converted_test_array[:,-1],enet_grid.predict(converted_test_array[:,:-1]))


# In[36]:

prediction = enet_grid.predict(test_fin_array_converted)
save_submission('submission1ENET3.csv',indexes,prediction)


# This grid parameter set got RMSE=0.16393 and was ranked 1504.

# # Performing feature selection

# In[503]:

from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import f_regression
kbest = GenericUnivariateSelect(f_regression, mode='k_best', param=79).fit(converted_train_array[:,:-1], converted_train_array[:,-1])


# In[534]:

for idx in range(len(kbest.scores_)):
    if kbest.scores_[idx]<4:
        print columns[idx], kbest.scores_[idx]
kbest_result = [columns[idx] for idx in range(len(columns)-1) if kbest.scores_[idx]<2]
kbest_result_idx = [idx for idx in range(len(columns)-1) if kbest.scores_[idx]>=2]


# In[507]:

len(kbest_result_idx)


# Testing on the best so far regression model.

# In[39]:

enet_grid = ElasticNet(alpha=1, l1_ratio=0.5)
enet_grid.fit(converted_train_array[:,kbest_result_idx], converted_train_array[:,-1])
print eval_rmse(converted_test_array[:,-1], enet_grid.predict(converted_test_array[:,kbest_result_idx]))
prediction = enet_grid.predict(test_fin_array_converted[:,kbest_result_idx])
save_submission('submission1ENET4.csv',indexes,prediction)


# Conclusion: a little bit better performance on test, but worse on real test.

# # DRAFT

# In[40]:

def myscorer(estimator,X,y):
    return eval_rmse(estimator.predict(X),y)


# In[41]:

from sklearn.feature_selection import RFECV


# In[42]:

estimator = ElasticNet(alpha=1, l1_ratio=0.5)
selector = RFECV(estimator, step=1, cv=5, scoring=myscorer)
selector = selector.fit(converted_train_array[:,:-1], converted_train_array[:,-1])


# In[43]:

selector.ranking_


# In[44]:

for idx in range(len(selector.ranking_)):
    if selector.ranking_[idx] > 59:
        print columns[idx], selector.ranking_[idx]


# In[45]:

from sklearn.feature_selection import SelectKBest, GenericUnivariateSelect, VarianceThreshold, SelectFwe
from sklearn.feature_selection import chi2, f_regression, f_classif
kbest = GenericUnivariateSelect(f_regression, mode='k_best', param=59).fit(converted_train_array[:,:-1], converted_train_array[:,-1])


# In[46]:

for idx in range(len(kbest.scores_)):
    if kbest.scores_[idx]<2:
        print columns[idx], kbest.scores_[idx]
kbest_result = [columns[idx] for idx in range(len(columns)-1) if kbest.scores_[idx]<2]


# perform regression without those

# In[48]:

fwe = SelectFwe(f_regression, alpha=0.7)
fwe.fit(converted_train_array[:,:-1], converted_train_array[:,-1])
for idx in range(len(columns)-1):
    if not idx in fwe.get_support(indices=True):
        print columns[idx]


# In[49]:

variance = VarianceThreshold(threshold=1)
variance.fit(converted_train_array[:,:-1])
print len(variance.get_support(indices=True))
for idx in range(len(columns)-1):
    if not idx in variance.get_support(indices=True):
        print columns[idx]
variance_result = [columns[idx] for idx in range(len(columns)-1) if not idx in variance.get_support(indices=True)]


# In[50]:

for col in variance_result:
    print col, len(pd.DataFrame(data)[col].value_counts())


# In[51]:

for column in columns:
    if column in variance_result and column in kbest_result:
        print column


# # XGBoost

# In[52]:

import sys
sys.path.append('/Volumes/HDD/Users/Andrey/xgboost_setup/xgboost/python-package/')
import xgboost as xgb


# In[53]:

import time


# In[300]:

def scorer_rmse(estimator, X, y):
    return 1-eval_rmse(estimator.predict(X),y)


# In[500]:

def find_xgb_model(x_train, y_train, x_test, y_test, x_final, x_final_indexes=None, filename=None, goodset=True,
                  plot=True):
    t_start = time.clock()
    if not goodset:
        params = {'max_depth' : [5,8], 
                  'learning_rate' : [0.01,0.05,0.1,0.15],
                  'n_estimators' : [100, 200, 250],
                  'min_child_weight' : [1, 0.5],
                  'reg_alpha' : [0, 0.1, 0.5, 0.9, 1, 2, 5],
                  'reg_lambda' : [0.1, 0.5, 1],
                  'base_score' : [30000],
                  'colsample_bytree' : [0.1,0.5,1],
                  'colsample_bylevel' : [0.5,1]}
    else:
        params = {'max_depth' : [5], 
                  'learning_rate' : [0.1],
                  'n_estimators' : [100, 200, 250],
                  'min_child_weight' : [1],
                  'reg_alpha' : [0, 0.1, 0.5, 0.9, 1],
                  'reg_lambda' : [1],
                  'base_score' : [30000],
                  'colsample_bytree' : [0.1,0.5,1],
                  'colsample_bylevel' : [0.5,1]}
    xgbreg = xgb.XGBRegressor(silent=True, objective='reg:linear', nthread=-1, gamma=0, max_delta_step=0, 
                                  subsample=1, scale_pos_weight=1, seed=0, missing=None)
    cv = KFold(len(y_train), n_folds=5)
    clf = grid_search.GridSearchCV(xgbreg, params, cv=cv)
    clf.fit(x_train, y_train)
    print "RMSE on given test: ", eval_rmse(clf.predict(x_test), y_test)
    print 'RMSE on train: ', eval_rmse(clf.predict(x_train), y_train)
    prediction = clf.predict(x_final)
    save_submission(filename,indexes,prediction)
    t_finish = time.clock()
    print clf.best_params_
    print "Spent time: ", t_finish-t_start, "s"
    if plot:
        price_comparation_graph(y_test,clf.predict(x_test))


# In[501]:

find_xgb_model(converted_train_array[:,kbest_result_idx], converted_train_array[:,-1], 
               converted_test_array[:,kbest_result_idx], converted_test_array[:,-1],
               test_fin_array_converted[:,kbest_result_idx], indexes, 'xgb_run11_big.csv', goodset=False)


# In[461]:

xgbreg = xgb.XGBRegressor(reg_alpha=0.1, colsample_bytree=0.1, colsample_bylevel=1, learning_rate=0.1,
                          min_child_weight=1, silent=True, objective='reg:linear', nthread=-1, gamma=0, 
                          max_delta_step=0, subsample=1, scale_pos_weight=1, seed=0, n_estimators=250,
                          reg_lambda=1, base_score=30000, max_depth=5,missing=None)
xgbreg.fit(converted_train_array[:,kbest_result_idx], converted_train_array[:,-1])
print eval_rmse(xgbreg.predict(converted_train_array[:, kbest_result_idx]), converted_train_array[:,-1])


# In[502]:

find_xgb_model(train_array_onehot[:,:-1], train_array_onehot[:,-1], 
               test_array_onehot[:,:-1], test_array_onehot[:,-1],
               test_fin_array_onehot, indexes, 'xgb_run11_onehot2.csv', goodset=False)


# In[453]:

xgbreg = xgb.XGBRegressor(max_depth=5, learning_rate=0.1,n_estimators=290, min_child_weight=1,
                              reg_alpha=0.01, reg_lambda=0.08, base_score=30000,silent=True, objective='reg:linear', 
                              nthread=-1, gamma=0, max_delta_step=0, subsample=1, colsample_bytree=0.1, 
                              colsample_bylevel=0.5, scale_pos_weight=1, seed=0, missing=None)
xgbreg.fit(converted_train_array[:,kbest_result_idx], converted_train_array[:,-1])
print eval_rmse(xgbreg.predict(converted_test_array[:, kbest_result_idx]), converted_test_array[:,-1])
print eval_rmse(xgbreg.predict(converted_train_array[:, kbest_result_idx]), converted_train_array[:,-1])


# In[340]:

matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)
xgb.plot_importance(xgbreg, height=0.2, importance_type='weight')


# In[342]:

np.array(columns)[kbest_result_idx]


# In[284]:

prediction = xgbreg.predict(test_fin_array_converted[:,kbest_result_idx])
save_submission("xbreg_run9_kbest.csv",indexes,prediction)


# In[285]:

price_comparation_graph(converted_test_array[:,-1],xgbreg.predict(converted_test_array[:,kbest_result_idx]))


# # DRAFT FOR LOW/MED/HIGH PRICES

# In[170]:

converted_train_array_low = np.row_stack([x for x in converted_train_array if x[-1] < 100001])
converted_test_array_low = np.row_stack([x for x in converted_test_array if x[-1] < 100001])


# In[279]:

xgbreg_low = xgb.XGBRegressor(max_depth=100, learning_rate=0.1,n_estimators=100, min_child_weight=1,
                              reg_alpha=0, reg_lambda=1, base_score=0,silent=True, objective='reg:linear', 
                              nthread=-1, gamma=0, max_delta_step=0, subsample=1, colsample_bytree=0.1, 
                              colsample_bylevel=0.5, scale_pos_weight=1, seed=0, missing=None)
xgbreg_low.fit(converted_train_array_low[:,kbest_result_idx], converted_train_array_low[:,-1])
print eval_rmse(xgbreg_low.predict(converted_test_array_low[:,kbest_result_idx]), converted_test_array_low[:,-1])
print eval_rmse(xgbreg.predict(converted_test_array_low[:,kbest_result_idx]), converted_test_array_low[:,-1])
print eval_rmse(xgbreg_low.predict(converted_test_array[:,kbest_result_idx]), converted_test_array[:,-1])
price_comparation_graph(converted_test_array_low[:,-1],xgbreg_low.predict(converted_test_array_low[:,kbest_result_idx]))
price_comparation_graph(converted_test_array[:,-1],xgbreg_low.predict(converted_test_array[:,kbest_result_idx]))


# In[412]:

converted_train_array_med = np.row_stack([x for x in converted_train_array if x[-1] > 100000 and x[-1] <200001])
converted_test_array_med = np.row_stack([x for x in converted_test_array if x[-1] > 100000 and x[-1] < 200001])
xgbreg_med = xgb.XGBRegressor(max_depth=100, learning_rate=0.1,n_estimators=100, min_child_weight=1,
                              reg_alpha=0, reg_lambda=1, base_score=0,silent=True, objective='reg:linear', 
                              nthread=-1, gamma=0, max_delta_step=0, subsample=1, colsample_bytree=0.1, 
                              colsample_bylevel=0.5, scale_pos_weight=1, seed=0, missing=None)
xgbreg_med.fit(converted_train_array_med[:,kbest_result_idx], converted_train_array_med[:,-1])
print eval_rmse(xgbreg_med.predict(converted_test_array_med[:,kbest_result_idx]), converted_test_array_med[:,-1])
print eval_rmse(xgbreg.predict(converted_test_array_med[:,kbest_result_idx]), converted_test_array_med[:,-1])
print eval_rmse(xgbreg_med.predict(converted_test_array[:,kbest_result_idx]), converted_test_array[:,-1])
price_comparation_graph(converted_test_array_med[:,-1],xgbreg_med.predict(converted_test_array_med[:,kbest_result_idx]))
price_comparation_graph(converted_test_array[:,-1],xgbreg_med.predict(converted_test_array[:,kbest_result_idx]))


# In[282]:

converted_train_array_high = np.row_stack([x for x in converted_train_array if x[-1] > 200000])
converted_test_array_high = np.row_stack([x for x in converted_test_array if x[-1] > 200000])
xgbreg_high = xgb.XGBRegressor(max_depth=100, learning_rate=0.1,n_estimators=100, min_child_weight=1,
                              reg_alpha=0, reg_lambda=1, base_score=0,silent=True, objective='reg:linear', 
                              nthread=-1, gamma=0, max_delta_step=0, subsample=1, colsample_bytree=0.1, 
                              colsample_bylevel=0.5, scale_pos_weight=1, seed=0, missing=None)
xgbreg_high.fit(converted_train_array_high[:,kbest_result_idx], converted_train_array_high[:,-1])
print eval_rmse(xgbreg_high.predict(converted_test_array_high[:,kbest_result_idx]), converted_test_array_high[:,-1])
print eval_rmse(xgbreg.predict(converted_test_array_high[:,kbest_result_idx]), converted_test_array_high[:,-1])
print eval_rmse(xgbreg_high.predict(converted_test_array[:,kbest_result_idx]), converted_test_array[:,-1])
price_comparation_graph(converted_test_array_high[:,-1],xgbreg_high.predict(converted_test_array_high[:,kbest_result_idx]))
price_comparation_graph(converted_test_array[:,-1],xgbreg_high.predict(converted_test_array[:,kbest_result_idx]))


# In[283]:

print eval_rmse(np.concatenate([converted_test_array_low[:,-1],
                                       converted_test_array_med[:,-1],
                                       converted_test_array_high[:,-1]]),
                        np.concatenate([xgbreg_low.predict(converted_test_array_low[:,kbest_result_idx]),
                                        xgbreg_med.predict(converted_test_array_med[:,:-1]), 
                                        xgbreg_high.predict(converted_test_array_high[:,kbest_result_idx])]))
print eval_rmse(np.concatenate([converted_test_array_low[:,-1],
                                       converted_test_array_med[:,-1],
                                       converted_test_array_high[:,-1]]),
                        np.concatenate([xgbreg.predict(converted_test_array_low[:,kbest_result_idx]),
                                        xgbreg.predict(converted_test_array_med[:,kbest_result_idx]), 
                                        xgbreg.predict(converted_test_array_high[:,kbest_result_idx])]))
price_comparation_graph(np.concatenate([converted_test_array_low[:,-1],
                                       converted_test_array_med[:,-1],
                                       converted_test_array_high[:,-1]]),
                        np.concatenate([xgbreg_low.predict(converted_test_array_low[:,kbest_result_idx]),
                                        xgbreg_med.predict(converted_test_array_med[:,:-1]), 
                                        xgbreg_high.predict(converted_test_array_high[:,kbest_result_idx])]))


# In[211]:

price_comparation_graph(converted_test_array[:,-1],xgbreg.predict(converted_test_array[:,kbest_result_idx]))


# In[216]:

np.concatenate([[1,2,3], [4,5,6]]) - [1,1,1,1,1,1]


# In[223]:

plt.plot(xgbreg_low.predict(converted_test_array[:,:-1]) - xgbreg.predict(converted_test_array[:,kbest_result_idx]))
plt.plot(xgbreg_med.predict(converted_test_array[:,:-1]) - xgbreg.predict(converted_test_array[:,kbest_result_idx]))
plt.plot(xgbreg_high.predict(converted_test_array[:,kbest_result_idx]) - xgbreg.predict(converted_test_array[:,kbest_result_idx]))


# In[245]:

plt.figure(figsize=(60,20))
plt.plot(xgbreg_low.predict(converted_test_array[:,:-1]),'bo')
plt.plot(xgbreg_med.predict(converted_test_array[:,:-1]),'go')
plt.plot(xgbreg_high.predict(converted_test_array[:,kbest_result_idx]),'ro')
plt.plot(xgbreg.predict(converted_test_array[:,kbest_result_idx]),'yo')
plt.plot(converted_test_array[:,-1],'co')
plt.xticks(np.arange(0,300,1))
plt.grid()


# In[264]:

preds = np.concatenate([xgbreg.predict(converted_test_array_low[:,kbest_result_idx]),
                        xgbreg.predict(converted_test_array_med[:,kbest_result_idx]), 
                        xgbreg.predict(converted_test_array_high[:,kbest_result_idx])])
preds_cat = np.concatenate([xgbreg_low.predict(converted_test_array_low[:,:-1]),
                        xgbreg_med.predict(converted_test_array_med[:,:-1]), 
                        xgbreg_high.predict(converted_test_array_high[:,kbest_result_idx])])

preds_low = np.concatenate([xgbreg_low.predict(converted_test_array_low[:,:-1]),
                        xgbreg_low.predict(converted_test_array_med[:,:-1]), 
                        xgbreg_low.predict(converted_test_array_high[:,:-1])])
preds_med = np.concatenate([xgbreg_med.predict(converted_test_array_low[:,:-1]),
                        xgbreg_med.predict(converted_test_array_med[:,:-1]), 
                        xgbreg_med.predict(converted_test_array_high[:,:-1])])
preds_high = np.concatenate([xgbreg_high.predict(converted_test_array_low[:,kbest_result_idx]),
                        xgbreg_high.predict(converted_test_array_med[:,kbest_result_idx]), 
                        xgbreg_high.predict(converted_test_array_high[:,kbest_result_idx])])

truth = np.concatenate([converted_test_array_low[:,-1],
                            converted_test_array_med[:,-1],
                            converted_test_array_high[:,-1]])
low = len(converted_test_array_low)
med = len(converted_test_array_med)
high = len(converted_test_array_high)
tlow = 0
tmed = 0
thigh = 0
for idx in range(len(converted_test_array)):
    if truth[idx] < 100001:
        if preds[idx] < 100001:
            tlow = tlow + 1
        else:
            print "exc. low: ", preds[idx], preds_med[idx], truth[idx]
    else:
        if truth[idx] < 200001:
            if preds[idx] < 200001:
                tmed = tmed + 1
            else:
                print "exc. med: ", preds[idx], preds_high[idx], truth[idx]
        else:
            if truth[idx] > 200000:
                if preds[idx] > 200000:
                    thigh = thigh + 1
                else:
                    print "exc. high: ", preds[idx], preds_med[idx], truth[idx]
print low, tlow
print med, tmed
print high, thigh


# In[414]:

preds_l_f = xgbreg_low.predict(converted_train_array[:,kbest_result_idx])
preds_m_f = xgbreg_med.predict(converted_train_array[:,kbest_result_idx])
preds_h_f = xgbreg_high.predict(converted_train_array[:,kbest_result_idx])

preds_f = xgbreg.predict(converted_train_array[:,kbest_result_idx])
print eval_rmse(preds_f, converted_train_array[:,-1]) 
price_comparation_graph(converted_train_array[:,-1],preds_f)

"""
for idx in range(len(preds_f)):
    if preds_f[idx] < 100001:
        preds_f[idx] = preds_l_f[idx]
    else:
        if preds_f[idx] < 200001:
            preds_f[idx] = preds_m_f[idx]
        else:
            preds_f[idx] = preds_h_f[idx]
print eval_rmse(preds_f, converted_test_array[:,-1])
#save_submission("xbreg_run8_kbest_cat.csv",indexes,preds_f)
price_comparation_graph(converted_test_array[:,-1],preds_f)
"""

train2 = np.column_stack([preds_l_f, preds_m_f, preds_h_f, converted_train_array[:,-1]])

xgb2 = xgb.XGBRegressor(max_depth=100, learning_rate=0.1,n_estimators=100, min_child_weight=1,
                              reg_alpha=0, reg_lambda=1, base_score=0,silent=True, objective='reg:linear', 
                              nthread=-1, gamma=0, max_delta_step=0, subsample=1, colsample_bytree=1, 
                              colsample_bylevel=1, scale_pos_weight=1, seed=0, missing=None)
xgb2.fit(train2[:,:-1], train2[:,-1])

preds_l_f = xgbreg_low.predict(converted_test_array[:,kbest_result_idx])
preds_m_f = xgbreg_med.predict(converted_test_array[:,kbest_result_idx])
preds_h_f = xgbreg_high.predict(converted_test_array[:,kbest_result_idx])
test2 = np.column_stack([preds_l_f, preds_m_f, preds_h_f, converted_test_array[:,-1]])

pred_add = xgb2.predict(test2[:,:-1])
price_comparation_graph(converted_test_array[:,-1],pred_add)


# In[362]:

train_class = np.zeros(converted_train_array[:,-1].shape)
test_class = np.zeros(converted_test_array[:,-1].shape)
for idx in range(len(converted_train_array[:,-1])):
    train_class[idx] = int(min(3,converted_train_array[idx,-1] // 100000 + 1))
for idx in range(len(converted_test_array[:,-1])):
    test_class[idx] = int(min(3,converted_test_array[idx,-1] // 100000 + 1))


# In[423]:

diff = preds - test_class
for idx in range(len(diff)):
    if diff[idx] !=0 :
        print preds[idx], converted_test_array[idx,-1], preds_l_f[idx], preds_m_f[idx], preds_h_f[idx]

preds_f = np.zeros(preds.shape)
for idx in range(len(diff)):
    if preds[idx]==1:
        preds_f[idx] = preds_l_f[idx]
    else:
        if preds[idx]==2:
            preds_f[idx] = preds_m_f[idx]
        else:
            preds_f[idx] = preds_h_f[idx]
print eval_rmse(preds_f, converted_test_array[:,-1])


# In[424]:

price_comparation_graph(converted_test_array[:,-1],preds_f)


# In[ ]:



