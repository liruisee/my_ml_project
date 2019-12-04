from sklearn_demo.line_model.line_classfy import (
    LinearDiscriminantAnalysisModel,
    SVCModel,
    SGDClassifierModel
)


if __name__ == '__main__':
    # 过滤一些告警信息
    import warnings
    warnings.filterwarnings('ignore')
    # 线性判别分析
    lda = LinearDiscriminantAnalysisModel()
    lda.load_data()
    lda.train()
    lda.predict()
    lda.estimate_model()

    # 支持向量机
    svc = SVCModel()
    svc.load_data()
    svc.train()
    svc.predict()
    svc.estimate_model()

    # 随机梯度下降
    sgd = SGDClassifierModel()
    sgd.load_data()
    sgd.train()
    sgd.predict()
    sgd.estimate_model()
