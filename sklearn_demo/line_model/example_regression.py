from sklearn_demo.line_model.line_regression import (
    LinearRegressionModel,
    RidgeModel,
    RidgeCVModel,
    LassoModel,
    LassoCVModel,
    LassoLarsCVModel,
    SVRModel,
    SGDRegressorModel
)


if __name__ == '__main__':
    # 过滤一些告警信息
    import warnings
    warnings.filterwarnings('ignore')
    # 线性回归
    ln = LinearRegressionModel()
    # 数据加载
    ln.load_data()
    # 模型训练
    ln.train()
    # 模型预测
    ln.predict()
    # 模型评估
    ln.estimate_model()
    print('')

    # 岭回归
    rm = RidgeModel()
    # 数据加载
    rm.load_data()
    # 模型训练
    rm.train()
    # 模型预测
    rm.predict()
    # 模型评估
    rm.estimate_model()
    print('')

    # 岭回归交叉验证
    rmcv = RidgeCVModel(alphas=[0.1, 1, 10])
    # 数据加载
    rmcv.load_data()
    # 模型训练
    rmcv.train()
    # 模型预测
    rmcv.predict()
    # 模型评估
    rmcv.estimate_model()
    print('')

    # lasso回归
    lm = LassoModel()
    # 数据加载
    lm.load_data()
    # 模型训练
    lm.train()
    # 模型预测
    lm.predict()
    # 模型评估
    lm.estimate_model()
    print('')

    lcvm = LassoCVModel(alphas=[0.1, 1, 10])
    # 数据加载
    lcvm.load_data()
    # 模型训练
    lcvm.train()
    # 模型预测
    lcvm.predict()
    # 模型评估
    lcvm.estimate_model()
    print('')

    llcvm = LassoLarsCVModel()
    # 数据加载
    llcvm.load_data()
    # 模型训练
    llcvm.train()
    # 模型预测
    llcvm.predict()
    # 模型评估
    llcvm.estimate_model()
    print('')

    svr = SVRModel()
    # 数据加载
    svr.load_data()
    # 模型训练
    svr.train()
    # 模型预测
    svr.predict()
    # 模型评估
    svr.estimate_model()
    print('')

    sgd = SGDRegressorModel()
    # 数据加载
    sgd.load_data()
    # 模型训练
    sgd.train()
    # 模型预测
    sgd.predict()
    # 模型评估
    sgd.estimate_model()
    print('')
