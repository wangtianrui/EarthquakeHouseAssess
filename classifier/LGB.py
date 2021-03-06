from classifier.MyModel import MyModel
import lightgbm as lgb
import numpy as np
import utils.scores as score


class LGB(MyModel):

    def __init__(self, params):
        super().__init__()

        """
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_error',
        'num_leaves': 200,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'lambda_l1': 0.6,
        'lambda_l2': 0.6,
        'verbose': 0
        """
        self.model = lgb.LGBMClassifier(boosting_type='gbdt', objective='muticlass', num_leaves=200, learning_rate=0.1,
                                        reg_lambda=0.6, num_class=4, metric='multi_error', n_estimators=1000)

    def train(self, X_train, y_train, X_test=None, y_test=None, step=100, log=False):
        self.model.fit(X_train, y_train)
        if log:
            random_index = np.random.choice(X_test.shape[0], 500)
            X = X_test[random_index]
            # print(y_predict[1])
            # self.scores(y_pred=y_predict, y_label=y_test[random_index])
            softmax = self.model.predict(X)
            print(softmax.shape)
            print("accuracy:%f", (score.accuracy_score(softmax, y_test[random_index])))


    def predict(self, X_test):
        return self.model.predict(X_test)

    def gridSearchCV(self, grid_params, X, y, cv=5):
        super().gridSearchCV(grid_params, X, y, cv)
