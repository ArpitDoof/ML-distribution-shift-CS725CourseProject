from src.data.data import get
from src.models.svmdecisiontree import build_svm, build_decision_tree

X_train, X_test, y_train, y_test = get(
    type="iid",
    random_state=42
)

# SVM
svm = build_svm(kernel="rbf")
svm.fit(X_train, y_train)

svm_prob = svm.predict_proba(X_test)[:, 1]

# Decision Tree
tree = build_decision_tree(max_depth=5)
tree.fit(X_train, y_train)

tree_prob = tree.predict_proba(X_test)[:, 1]

print("SVM probabilities:", svm_prob[:5])
print("Tree probabilities:", tree_prob[:5])

svm_probs = svm.predict_proba(X_test)[:, 1]
tree_probs = tree.predict_proba(X_test)[:, 1]

print("SVM shape:", svm_probs.shape)
print("Tree shape:", tree_probs.shape)

print("SVM probability range:", svm_probs.min(), svm_probs.max())
print("Tree probability range:", tree_probs.min(), tree_probs.max())