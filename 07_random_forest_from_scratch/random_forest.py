"""Random Forest From Scratch。
主線：資料 → 固定切分 → 比較樹數 → Validation 選模 → Test → 圖表。
Bootstrap 改變各樹的樣本；Random Feature Selection 改變各節點的候選欄位。
沿用 06 的切點搜尋、葉節點與單棵樹預測，不重複建立基礎功能。
"""
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# 找到上一章的資料夾，沿用已完成的核心功能。
tree_folder = (
    Path(__file__).resolve().parent.parent
    / "06_decision_tree_from_scratch"
)
sys.path.insert(0, str(tree_folder))

from decision_tree import find_best_threshold, make_leaf, predict


# ==================== 1. 實驗設定 ====================

CONFIG = {
    "data_seed": 42,
    "split_seed": 42,
    "forest_seed": 42,
    "n_samples": 200,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "max_features": 1,
    "max_depth": 10,
    "min_samples_split": 2,
    "n_estimators_values": (1, 10, 50),  # 分別比較的樹數量；每輪只傳入其中一個整數
}

# ==================== 2. Bootstrap Sampling ====================

def bootstrap_sampling(X, y, rng):
    """輸入配對樣本與產生器，回傳相同筆數的 Bootstrap 樣本與標籤。"""
    # 抽取與原資料相同筆數的編號，允許重複。
    sample_indices = rng.choice(len(X), size=len(X), replace=True)

    # X 與 y 必須使用同一組編號，保持樣本與標籤對應。
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]

    return X_sample, y_sample


# ==================== 3. 隨機選取候選 Features ====================

def select_feature_indices(n_features, max_features, rng):
    if (isinstance(max_features, (bool, np.bool_))
            or not isinstance(max_features, (int, np.integer))
            or not 1 <= max_features <= n_features):
        raise ValueError("max_features 必須介於 1 與 Feature 總數之間")

    feature_indices = rng.choice(
        n_features,
        size=max_features,
        replace=False,
    )

    return feature_indices

# ==================== 4. 在隨機候選欄位中找最佳切分 ====================

def find_best_split(X, y, max_features, rng):

    # 每次處理節點，重新抽選候選欄位；不是整棵樹固定同一欄。
    feature_indices = select_feature_indices(
        n_features=X.shape[1],
        max_features=max_features,
        rng=rng,
    )

    best_split = None
    best_score = np.inf

    # 其餘流程沿用 Decision Tree。
    for feature_index in feature_indices:
        x_column = X[:, feature_index]

        threshold, score = find_best_threshold(x_column, y)

        # 這一欄全部同值時，沒有可用切點。
        if threshold is None:
            continue

        if score < best_score:
            best_score = score
            best_split = {
                "feature_index": feature_index,
                "threshold": threshold,
                "weighted_gini": score,
            }

    return best_split


# ==================== 5. 建立一棵隨機樹 ====================

def build_random_tree(
    X, y, max_depth, min_samples_split, max_features, rng, depth=0,
):

    # ① 決定這個節點是否停止。
    if (
        depth >= max_depth
        or len(y) < min_samples_split
        or len(np.unique(y)) == 1
    ):
        return make_leaf(y)

    # ② 在隨機抽到的欄位中，尋找最佳切分。
    split = find_best_split(X, y, max_features, rng)

    # 抽到的欄位全部同值時停止；本版不額外抽其他欄位。
    # 允許零改善切分，仍由深度、樣本數及純度限制停止。
    if split is None:
        return make_leaf(y)

    # ③ 保存規則，並依規則分出左右資料。
    feature_index = split["feature_index"]
    threshold = split["threshold"]

    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    node = {
        "is_leaf": False,
        "feature_index": feature_index,
        "threshold": threshold,
    }

    # ④ 左右子集各自繼續建樹，並在各節點重新抽選欄位。
    node["left"] = build_random_tree(
        X[left_mask],
        y[left_mask],
        max_depth,
        min_samples_split,
        max_features,
        rng,
        depth=depth + 1,
    )

    node["right"] = build_random_tree(
        X[right_mask],
        y[right_mask],
        max_depth,
        min_samples_split,
        max_features,
        rng,
        depth=depth + 1,
    )

    return node

# ==================== 6. 訓練森林 ====================

def fit_random_forest(
    X,
    y,
    n_estimators,
    max_depth,
    min_samples_split,
    max_features,
    seed,
):
    if (isinstance(n_estimators, (bool, np.bool_))
            or not isinstance(n_estimators, (int, np.integer))
            or n_estimators < 1):
        raise ValueError("n_estimators 必須是正整數")
    # 每次訓練重設 Seed，讓不同樹數的實驗具有相同的前段森林。
    rng = np.random.default_rng(seed)
    trees = []

    for _ in range(n_estimators):
        X_sample, y_sample = bootstrap_sampling(X, y, rng)

        tree = build_random_tree(
            X_sample,
            y_sample,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            rng=rng,
        )

        trees.append(tree)

    return trees

# ==================== 7. 森林預測與多數決 ====================
def predict_random_forest(X_new, trees):
    """輸入新樣本與森林，回傳每筆樣本的多數決類別，形狀為 (n_samples,)。"""
    if len(trees) == 0:
        raise ValueError("森林至少需要一棵樹才能預測")
    # ① 每棵樹對全部新資料預測。
    tree_predictions = np.array([
        predict(X_new, tree)
        for tree in trees
    ])
    # tree_predictions 的形狀：
    # (樹的數量, 樣本數量)

    predictions = []

    # ② 每次取出同一筆樣本，在所有樹得到的答案。
    for votes in tree_predictions.T:
        counts = np.bincount(votes, minlength=2)
        # 同票時 argmax 選較小的類別編號。
        prediction = np.argmax(counts)
        predictions.append(prediction)

    return np.array(predictions, dtype=np.int64)


# ==================== 8. 主線實驗 ====================

def main():

    # ① 重現圓形資料。
    rng_data = np.random.default_rng(CONFIG["data_seed"])
    X = rng_data.uniform(-2, 2, size=(CONFIG["n_samples"], 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.44).astype(np.int64)

    # ② 固定切分，讓所有模型使用相同資料。
    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(len(X))

    n_train = int(len(X) * CONFIG["train_ratio"])
    n_validation = int(len(X) * CONFIG["validation_ratio"])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train:n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_validation, y_validation = X[validation_indices], y[validation_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print("Train：", X_train.shape, y_train.shape)
    print("Validation：", X_validation.shape, y_validation.shape)
    print("Test：", X_test.shape, y_test.shape)

    # ==================== 比較森林中的樹數量 ====================

    results = {}

    # 候選 tuple → 每輪取一個整數 → 訓練 → 評估 → 保存。
    for n_estimators in CONFIG["n_estimators_values"]:
        trees = fit_random_forest(
            X_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=CONFIG["max_depth"],
            min_samples_split=CONFIG["min_samples_split"],
            max_features=CONFIG["max_features"],
            seed=CONFIG["forest_seed"],
        )

        assert len(trees) == n_estimators

        train_predictions = predict_random_forest(X_train, trees)
        validation_predictions = predict_random_forest(
            X_validation,
            trees,
        )

        assert train_predictions.shape == y_train.shape
        assert validation_predictions.shape == y_validation.shape

        train_accuracy = np.mean(train_predictions == y_train)
        validation_accuracy = np.mean(
            validation_predictions == y_validation
        )

        results[n_estimators] = {
            "trees": trees,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        }

        print(
            f"Trees={n_estimators}："
            f"Train ACC={train_accuracy:.4f}，"
            f"Validation ACC={validation_accuracy:.4f}"
        )

    # ==================== Validation 選模 ====================

    # 依成績選擇；同分時保留候選順序較早的樹數。
    def get_validation_accuracy(n_estimators):
        return results[n_estimators]["validation_accuracy"]


    selected_n_estimators = max(
        results,
        key=get_validation_accuracy,
    )

    selected_result = results[selected_n_estimators]
    selected_trees = selected_result["trees"]

    print("選定樹數：", selected_n_estimators)
    print(
        "選定 Validation Accuracy：",
        selected_result["validation_accuracy"],
    )

    # ==================== 最終 Test 評估 ====================
    # 沿用選定森林，不以 Test 成績重新調整設定。

    test_predictions = predict_random_forest(
        X_test,
        selected_trees,
    )

    test_accuracy = np.mean(test_predictions == y_test)

    print("Test Accuracy：", test_accuracy)

    assert test_predictions.shape == y_test.shape


    # ==================== 選定森林的決策邊界 ====================

    grid_axis = np.linspace(-2, 2, 150)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)

    grid_points = np.column_stack((
        grid_x.ravel(),
        grid_y.ravel(),
    ))

    grid_predictions = predict_random_forest(
        grid_points,
        selected_trees,
    )
    grid_labels = grid_predictions.reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.contourf(
        grid_x,
        grid_y,
        grid_labels,
        levels=[-0.5, 0.5, 1.5],
        colors=["tab:blue", "tab:orange"],
        alpha=0.25,
    )

    for label, color, name in (
        (0, "tab:blue", "Train Normal"),
        (1, "tab:orange", "Train Abnormal"),
    ):
        members = X_train[y_train == label]

        ax.scatter(
            members[:, 0],
            members[:, 1],
            color=color,
            edgecolors="black",
            label=name,
        )

    true_boundary = plt.Circle(
        (0, 0),
        radius=1.2,
        fill=False,
        color="black",
        linestyle="--",
        label="True boundary",
    )
    ax.add_patch(true_boundary)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(
        f"Random Forest ({selected_n_estimators} trees)"
    )
    ax.legend(loc="upper right")

    fig.tight_layout()

    output_path = (
        Path(__file__).resolve().parent
        / "random_forest_boundary.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print("決策邊界圖已儲存：", output_path)

    # ==================== 樹數量與準確率比較 ====================

    tree_counts = list(CONFIG["n_estimators_values"])

    # 依相同順序取出成績，確保橫軸與縱軸一一對應。
    train_accuracies = [
        results[count]["train_accuracy"]
        for count in tree_counts
    ]

    validation_accuracies = [
        results[count]["validation_accuracy"]
        for count in tree_counts
    ]

    fig_scores, ax_scores = plt.subplots(figsize=(8, 5))

    ax_scores.plot(
        tree_counts,
        train_accuracies,
        marker="o",
        label="Training Accuracy",
    )

    ax_scores.plot(
        tree_counts,
        validation_accuracies,
        marker="o",
        label="Validation Accuracy",
    )

    ax_scores.set_xticks(tree_counts)
    ax_scores.set_ylim(0, 1.05)
    ax_scores.set_xlabel("Number of Trees")
    ax_scores.set_ylabel("Accuracy")
    ax_scores.set_title("Random Forest Tree Count Comparison")
    ax_scores.legend()
    ax_scores.grid(alpha=0.2)

    fig_scores.tight_layout()

    comparison_path = (
        Path(__file__).resolve().parent
        / "random_forest_tree_count_comparison.png"
    )

    fig_scores.savefig(
        comparison_path,
        dpi=150,
        bbox_inches="tight",
    )

    print("樹數比較圖已儲存：", comparison_path)


    plt.close("all")


if __name__ == "__main__":
    main()
