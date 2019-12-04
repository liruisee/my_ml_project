from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, SGDRegressor
from base.regression_model import RegressionModel
from utils.check_field import check_kwargs
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# 架子啊数据的基类
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
class LinearRegressionModel(LoadDataModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_model = LinearRegression()

    # 训练模型
    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        # 权重赋值
        self.coef_ = self.init_model.coef_
        # 偏置赋值
        self.intercept_ = self.init_model.intercept_
        return self.init_model

    # 预测模型
    def predict(self):
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        self.pre_data = self.model.predict(self.x_test)


# 岭回归，当样本量较少时适用
class RidgeModel(LoadDataModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 如果传入的alpha，则使用传入的alpha
        if 'alpha' in kwargs:
            self.init_model = Ridge(alpha=kwargs['alpha'])
        # 如果未传入alpha，取sklearn框架的默认值1
        else:
            self.init_model = Ridge()

    # 训练模型
    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        # 权重赋值
        self.coef_ = self.init_model.coef_
        # 偏置赋值
        self.intercept_ = self.init_model.intercept_
        return self.init_model

    # 预测模型
    def predict(self):
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        self.pre_data = self.model.predict(self.x_test)


# CV岭回归，当样本量较少时适用
class RidgeCVModel(LoadDataModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        need_keys = {'alphas'}
        check_kwargs(kwargs, need_keys=need_keys)
        assert(type(kwargs['alphas']) is list)
        self.init_model = RidgeCV(alphas=kwargs['alphas'])
        # 岭回归的lambda值列表，训练模型会尝试所有的可能，把其中最好结果的值记录下来，作为自己的alpha，更新到模型的alpha_属性上
        self.alphas = kwargs['alphas']
        # 模型
        self.alpha_ = None

    # 训练模型
    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        # 权重赋值
        self.coef_ = self.init_model.coef_
        # 偏置赋值
        self.intercept_ = self.init_model.intercept_
        # lambda参数赋值
        self.alpha_ = self.init_model.alpha_
        return self.init_model

    # 预测模型
    def predict(self):
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        self.pre_data = self.model.predict(self.x_test)

    def estimate_model(self):
        super().estimate_model()
        print(self.__class__.__name__, f'alpha列表{self.alphas}中效果最好的alpha_为：', self.alpha_)


# lasso回归
class LassoModel(LoadDataModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 如果传入的alpha，则使用传入的alpha
        if 'alpha' in kwargs:
            self.init_model = Lasso(alpha=kwargs['alpha'])
        # 如果未传入alpha，取sklearn框架的默认值1
        else:
            self.init_model = Lasso()

    # 训练模型
    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        # 权重赋值
        self.coef_ = self.init_model.coef_
        # 偏置赋值
        self.intercept_ = self.init_model.intercept_
        return self.init_model

    # 预测模型
    def predict(self):
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        self.pre_data = self.model.predict(self.x_test)


# CV lasso回归
class LassoCVModel(LoadDataModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        need_keys = {'alphas'}
        check_kwargs(kwargs, need_keys=need_keys)
        assert(type(kwargs['alphas']) is list)
        self.init_model = LassoCV(alphas=kwargs['alphas'])
        # 岭回归的lambda值列表，训练模型会尝试所有的可能，把其中最好结果的值记录下来，作为自己的alpha，更新到模型的alpha_属性上
        self.alphas = kwargs['alphas']
        # 模型
        self.alpha_ = None

    # 训练模型
    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        # 权重赋值
        self.coef_ = self.init_model.coef_
        # 偏置赋值
        self.intercept_ = self.init_model.intercept_
        # lambda参数赋值
        self.alpha_ = self.init_model.alpha_
        return self.init_model

    # 预测模型
    def predict(self):
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        self.pre_data = self.model.predict(X=self.x_test)

    def estimate_model(self):
        super().estimate_model()
        print(self.__class__.__name__, f'alpha列表{self.alphas}中效果最好的alpha_为：', self.alpha_)


# lasso回归
class LassoLarsCVModel(LoadDataModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_model = LassoLarsCV()

    # 训练模型
    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        # 权重赋值
        self.coef_ = self.init_model.coef_
        # 偏置赋值
        self.intercept_ = self.init_model.intercept_
        return self.init_model

    # 预测模型
    def predict(self):
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        self.pre_data = self.model.predict(X=self.x_test)


# 支持向量机回归
class SVRModel(LoadDataModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_model = SVR()

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
        self.pre_data = self.model.predict(X=self.x_test)


class SGDRegressorModel(LoadDataModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_model = SGDRegressor()

    # 加载数据，选择了波士顿的房价数据，并采用留出法划分训练集和测试集
    def load_data(self):
        if self.data is None:
            self.data = load_boston()
        x, y = self.data.data, self.data.target
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        scaler = StandardScaler()
        scaler.fit(x_train)
        self.x_train = scaler.transform(x_train)
        self.x_test = scaler.transform(x_test)
        self.y_train = y_train
        self.y_test = y_test

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
        self.pre_data = self.model.predict(X=self.x_test)
