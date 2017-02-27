import pandas as pd
from sklearn.metrics import log_loss
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression


training_data = pd.read_csv('numerai_training_data.csv')
tournament_data = pd.read_csv('numerai_tournament_data.csv')

# better validation method!
train, test = cross_validation.train_test_split(training_data,
                                                test_size=0.7,
                                                random_state=0)

features = list(train.columns[:-1])

penalties = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001]
best_penalty = 1.0
best_logloss = 1.0

for penalty in penalties:
    model = LogisticRegression(C=penalty, n_jobs=-1, random_state=0)
    model.fit(train[features], train['target'])
    test_predictions = model.predict_proba(test[features])

    logloss = log_loss(test['target'], test_predictions)
    print("Test Log Loss %f with penalty %f" % (logloss, penalty))

    if logloss <= best_logloss:
        best_penalty = penalty
        best_logloss = logloss

model = LogisticRegression(C=best_penalty, n_jobs=-1, random_state=0)
model.fit(training_data[features], training_data['target'])
test_predictions = model.predict_proba(training_data[features])
logloss = log_loss(training_data['target'], test_predictions)
print("Fully Trained Log Loss %f with penalty %f" % (logloss, best_penalty))

tournament_predictions = model.predict_proba(tournament_data[features])

result = tournament_data
result['probability'] = tournament_predictions[:,1]

result.to_csv("predictions/lr_predictions.csv",
              columns=('t_id', 'probability'),
              index=None)