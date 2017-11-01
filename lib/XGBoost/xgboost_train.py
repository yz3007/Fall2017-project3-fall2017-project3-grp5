import pandas as pd
import xgboost as xgb
import pickle

# --------------------
# --- Loading Data ---
# --------------------
feature = pd.read_csv("../input/sift_train.csv", header=0)
label_train = pd.read_csv("../input/label_train.csv")
feature_new = pd.read_csv("../input/feature.csv", header=None)

feature.columns = ['image']+ ['x'+str(i+1) for i in range(5000)]
label_train.columns = ['image', 'label']
feature_new.columns = ['image']+ ['f'+str(i+1) for i in range(960)]
feature_new['image'] = feature_new['image'].apply(lambda x: int(x.split(".")[0].split('_')[1]))

feature_all = pd.concat([feature, feature_new.drop("image", axis=1)], axis=1)
X = feature_all.drop("image", axis=1)
y = label_train['label']
print("Loaded Data.")

# ------------------
# --- Split Data ---
# ------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

# -----------------------------------
# --- Train Model: Baseline Model ---
# -----------------------------------


model_baseline = xgb.XGBClassifier(learning_rate = 0.1
                        , objective = 'multi:softmax'
			, n_estimators = 140
                        , max_depth = 1
                        , min_child_weight = 0
                        , subsample = 1
                        , colsample_bytree = 1
                        , gamma = 0
                        , seed = 1234
                        , nthread = -1)
model_baseline.fit(X_train, y_train)


# -------------------
# --- Tuned Model ---
# -------------------

xgb_BO_scores = pd.read_csv("../output/xgb_BO_5960f_scores.csv")
params = dict()
params['max_depth'] = int(xgb_BO_scores['max_depth'][0])
params['min_child_weight'] = int(xgb_BO_scores['min_child_weight'][0])
params['colsample_bytree'] = xgb_BO_scores['colsample_bytree'][0]
params['subsample'] = xgb_BO_scores['subsample'][0]
params['gamma'] = xgb_BO_scores['gamma'][0]


model_tuned = xgb.XGBClassifier(learning_rate = 0.1
                        , objective = 'multi:softmax'
                        , n_estimators = 140
                        , max_depth = params['max_depth']
                        , min_child_weight = params['min_child_weight']
                        , subsample = params['subsample']
                        , colsample_bytree = params['colsample_bytree']
                        , gamma = params['gamma']
                        , seed = 1234
                        , nthread = -1
                       )

model_tuned.fit(X_train, y_train)


# save the baseline model
filename = '../model/model_baseline.sav'
pickle.dump(model_baseline, open(filename, 'wb'))
# save the tuned model
filename = '../model/model_tuned.sav'
pickle.dump(model_tuned, open(filename, 'wb'))
