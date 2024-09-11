from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 创建k折交叉验证对象
'''
n_splits:将数据集分成几折
shuffle:随机打乱数据集
random_state:随机种子
'''
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 创建随机森林分类器
clf = RandomForestClassifier()

# 训练和测试模型
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 运用clf进行训练
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")