
import pandas as pd
import xgboost as xgb

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
y = label_train.drop("image", axis=1)

# ------------------
# --- Split Data ---
# ------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

# ---------------------
# --- Model Tuning ---
# ---------------------
from bayes_opt import BayesianOptimization

xg_train = xgb.DMatrix(X_train, label=y_train)  
xg_test = xgb.DMatrix(X_test, label=y_test)

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma):
    params = dict()
    params['objective'] = 'multi:softmax'
    params['num_class'] = 3
    params['eta'] = 0.1
    params['max_depth'] = int(max_depth)   
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = colsample_bytree
    params['subsample'] = subsample
    params['gamma'] = gamma
    params['verbose_eval'] = True    


    cv_result = xgb.cv(params, xg_train,
                       num_boost_round=100000,
                       nfold=5,
                       metrics={'merror'},
                       seed=1234,
                       callbacks=[xgb.callback.early_stop(50)])

    return -cv_result['test-merror-mean'].min()


xgb_BO = BayesianOptimization(xgb_evaluate, 
                             {'max_depth': (1, 10),
                              'min_child_weight': (0.5, 5),
                              'colsample_bytree': (0.01, 0.5),
                              'subsample': (0.1, 0.5),
                              'gamma': (0.01, 0.3)
                             }
                            )

xgb_BO.maximize(init_points=5, n_iter=20)

xgb_BO_scores = pd.DataFrame(xgb_BO.res['all']['params'])
xgb_BO_scores['score'] = pd.DataFrame(xgb_BO.res['all']['values'])
xgb_BO_scores = xgb_BO_scores.sort_values(by='score',ascending=False)

# save the tuning results
xgb_BO_scores.to_csv("../output/xgb_BO_5960f_scores.csv", index=False)
