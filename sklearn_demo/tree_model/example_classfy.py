from sklearn_demo.tree_model.tree_classfy import RandomForestModel, AdaBoostModel, GbdtModel


if __name__ == '__main__':
    # 随机森林
    model = RandomForestModel()
    model.load_data()
    model.train()
    model.predict()
    model.estimate_model()

    # ada boost
    model = AdaBoostModel()
    model.load_data()
    model.train()
    model.predict()
    model.estimate_model()

    # gbdt梯度提升树
    model = GbdtModel()
    model.load_data()
    model.train()
    model.predict()
    model.estimate_model()
