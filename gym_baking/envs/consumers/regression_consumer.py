import sklearn
from sklearn.svm import SVR as Estimator
from gym_baking.envs.consumers.base_consumer import BaseConsumer
from sklearn.metrics import r2_score
from gym_baking.envs.consumers.utils import *


class RegressionConsumer(BaseConsumer):
    def __init__(self, config):
        super().__init__(config)
        self.top_n_items = config['top_n_items']
        self.data_path = config['data_path']
        self.opening_time = get_min_of_day(config['opening_time'])
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

        self.data = load_data(self.data_path)
        self.data, self.X, self.y = preprocess_data(self.data, self.top_n_items, self.opening_time)
        self.X, self.y = reduce_sparsity(self.top_n_items, self.X, self.y)
        self.fit_model()

    def make_orders(self, inventory_products, order_queue, timestep):
        order = np.round(self.model.predict([[timestep]]))[0]

        # conform result to correct format
        return format_order_result(order)

    def fit_model(self):
        self.model = VectorRegression(Estimator(gamma='auto'))
        self.model.fit(self.X, self.y)

    def test_model(self):
        predictions = self.model.predict(self.X[0])
        print("Model accuracy: {}".format(r2_score(self.y[0], predictions)))


# Reference:
# http://stats.stackexchange.com/questions/153853/regression-with-scikit-learn-with-multiple-outputs-svr-or-gbm-possible
class VectorRegression(sklearn.base.BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = None

    def fit(self, X, y):
        # Fit a separate regressor for each column of y
        self.estimators_ = [sklearn.base.clone(self.estimator).fit(X[i], y[i])
                            for i in range(len(X))]
        return self

    def predict(self, X):
        # Join regressors' predictions
        res = [est.predict(X)[:, np.newaxis] for est in self.estimators_]
        return np.hstack(res)
