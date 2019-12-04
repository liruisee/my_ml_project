from sklearn.ensemble import GradientBoostingRegressor
from base.regression_model import RegressionModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


# 数据加载基类
class LoadDataModel(RegressionModel):

    # 加载数据，选择了波士顿的房价数据，并采用留出法划分训练集和测试集
    def load_data(self):
        if self.data is None:
            self.data = load_boston()
        x, y = self.data.data, self.data.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y)

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


# 线性回归，最基础的回归算法
class GbdtModel(LoadDataModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=1,
            random_state=0,
            loss='ls'
        )

    # 训练模型
    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        # 权重赋值
        # self.coef_ = self.init_model.coef_
        # 偏置赋值
        # self.intercept_ = self.init_model.intercept_
        return self.init_model

    # 预测模型
    def predict(self):
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        self.pre_data = self.model.predict(self.x_test)
