from sklearn_demo.tree_model.tree_regression import GbdtModel


if __name__ == '__main__':
    model = GbdtModel()
    model.load_data()
    model.train()
    model.predict()
    model.estimate_model()
