
# coding: utf-8

# In[1]:


from sklearn.model_selection import ShuffleSplit, cross_val_score
from hyperopt import tpe, hp, fmin, space_eval, Trials, STATUS_OK, STATUS_FAIL
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from DataLoader import DataLoader


data_loader = DataLoader()


X, X_train, X_test, y, y_train, y_test = data_loader.preprocess()



# In[2]:


#This function searches for the best classifier and parameters for the model. 

def hyperopt_tune(params):

    t = params['type']
    del params['type']

    """Lists and assigns the classifiers."""
    
    if t == 'naive_bayes':
        clf = BernoulliNB(**params)
    elif t == 'svm':
        clf = SVC(**params)
    elif t == 'dtree':
        clf = DecisionTreeClassifier(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    elif t == 'randomforest':            
        clf = RandomForestClassifier(**params)
    elif t == 'logistic regression':
        clf = LogisticRegression(**params)
    else:
        return 0
    return cross_val_score(clf, X, y).mean()
    
    """
    Creates a dictionary space for the parameters of classifiers. Better check documentations for the parameters.
    """
 
space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
        'alpha': hp.uniform('alpha', 0.0, 2.0)
    },
    {
        'type': 'svm',
        'C': hp.uniform('C', 0, 10.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20.0),
        'random_state': hp.choice('random_state',[42])
        
    },
    {
        'type': 'randomforest',
        'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,5)),
        'n_estimators': hp.choice('n_estimators', range(1,20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
     
        
    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('knn_n_neighbors', range(1,50))
    },
    
    {
        'type': 'logistic regression',
        'penalty': hp.choice('penalty', ['l2', 'l1']),
        'C': hp.uniform('C_space', (0.3, 0.7)),
        'random_state': hp.choice('rand_state', [42])
    }
    
])



count = 0
best = 0

def f(params):
    global best, count
    count += 1
    acc = hyperopt_tune(params.copy())
    if acc > best:
        print('new best:', acc, 'using', params['type'])
        best = acc
    if count % 50 == 0:
        print ('iters:', count, ', acc:', acc, 'using', params)
    return {'loss': -acc, 'status': STATUS_OK}
    return best
   
trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)

print ('best:')
print (best)


params=space_eval(space, best)

t = params['type'] 

del params['type']

if t == 'naive_bayes':
        clf = BernoulliNB(**params)
elif t == 'svm':
        clf = SVC(**params)
elif t == 'dtree':
        clf = DecisionTreeClassifier(**params)
elif t == 'knn':
        clf = KNeighborsClassifier(**params)
elif t == 'randomforest':            
        clf = RandomForestClassifier(**params)
elif t == 'logistic regression':
        clf = LogisticRegression(**params)





