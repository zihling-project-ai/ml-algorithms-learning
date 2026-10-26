"""Decision Tree From Scratch：Gini → 候選切分 → 遞迴建樹 → 預測。
實驗：固定資料切分 → 比較深度 → Validation 選模 → Test → Baseline 與圖表。
輔助單層樹與小資料驗證另見 decision_tree_learning.py。
輸入假設：非空有限數值 Features、非負整數類別標籤。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ==================== 實驗設定 ====================

CONFIG = {
    "data_seed": 42,
    "split_seed": 42,
    "n_samples": 200,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "min_samples_split": 2,
    "max_depths": (1, 2, 3, 5, 10),
}


# ==================== 1. Gini：評估標籤混雜程度 ====================

def compute_gini(y):
    """
    ① 確認有資料
    ② 計算各類別數量
    ③ 數量轉成比例
    ④ 比例平方後加總，衡量集中程度
    ⑤ 用 1 減掉，得到混雜程度
    """
    if len(y) == 0:
        raise ValueError("空集合無法計算類別比例")

    counts = np.bincount(y, minlength=2)
    # y=[0, 0, 1, 1] → counts=[2, 2]

    proportions = counts / len(y)
    # [2/4, 2/4] → [0.5, 0.5]

    same_class_p = np.sum(proportions ** 2)
    # 0.5² + 0.5² = 0.5
    # 也可解讀為放回抽兩次，兩次同類的機率。

    gini = 1 - same_class_p
    # 1 - 0.5 = 0.5；全同類時則為 0。

    return gini


# ==================== 2. 單欄：找最佳 Threshold ====================

def find_best_threshold(x, y):
    # ① 排序去重，再取相鄰數值的中點。
    values = np.unique(x)
    thresholds = (values[:-1] + values[1:]) / 2

    best_threshold = None
    best_score = np.inf

    for threshold in thresholds:
        # ② 用 Feature 決定位置，再取對應標籤。
        left_mask = x <= threshold
        right_mask = ~left_mask

        y_left = y[left_mask]
        y_right = y[right_mask]

        # ③ 分別衡量左右混雜程度。
        left_gini = compute_gini(y_left)
        right_gini = compute_gini(y_right)

        # ④ 按樣本數比例加權，得到單一切分分數。
        score = (
            len(y_left) / len(y) * left_gini
            + len(y_right) / len(y) * right_gini
        )

        # ⑤ 只保存更低分的候選；同分保留先找到的。
        if score < best_score:
            best_score = score
            best_threshold = threshold

    # 全部 Feature 值相同時，沒有候選，回傳 None、inf。
    return best_threshold, best_score


# ==================== 3. 多欄：找最佳 Feature 與 Threshold ====================

def find_best_split(X, y):
    best_split = None
    best_score = np.inf

    for feature_index in range(X.shape[1]):
        # X[列, 欄]；冒號表示所有列。
        x_column = X[:, feature_index]

        threshold, score = find_best_threshold(x_column, y)

        # 此欄沒有可用切點，就繼續下一欄。
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


# ==================== 4. 葉節點：保存多數類別 ====================

def make_leaf(y):
    prediction = np.argmax(np.bincount(y)) # bincount 數票，再用 argmax 選類別

    return {
        "is_leaf": True,
        "prediction": prediction,
    }


# ==================== 5. 目前節點：決定停止或切分 ====================

def prepare_node(X, y):
    # ① 只有一種類別，直接保存答案。
    if len(np.unique(y)) == 1:
        return make_leaf(y)

    # ② 標籤仍混合，才尋找切分。
    split = find_best_split(X, y)

    # ③ 所有 Features 都沒有可用切點。
    if split is None:
        return make_leaf(y)

    # 最佳切分仍無改善時停止；1e-12 是數值容差。
    # 此簡化規則可能略過需先做零改善切分才能發現的模式，例如 XOR。
    parent_gini = compute_gini(y)
    gini_decrease = parent_gini - split["weighted_gini"]

    if gini_decrease <= 1e-12:
        return make_leaf(y)

    # ⑤ 暫時保存候選規則，還沒有左右子節點。
    return {
        "is_leaf": False,
        "feature_index": split["feature_index"],
        "threshold": split["threshold"],
        "weighted_gini": split["weighted_gini"],
    }


# ==================== 6. 建樹：遞迴處理左右子集 ====================

def build_tree(X, y, max_depth, depth=0, min_samples_split=2):
    # ① 到達深度限制，不再切，保存多數類別。
    if depth >= max_depth:
        return make_leaf(y)

    if len(y) < min_samples_split:
        return make_leaf(y)

    # ② 檢查是否已純、是否有可用切分。
    node = prepare_node(X, y)

    if node["is_leaf"]:
        return node

    # ③ 用找到的規則分左右。
    feature_index = node["feature_index"]
    threshold = node["threshold"]

    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    # ④ 同一套建樹流程，分別處理左右子集。
    node["left"] = build_tree(
        X[left_mask],
        y[left_mask],
        max_depth,
        depth=depth + 1,
        min_samples_split=min_samples_split,
    )

    node["right"] = build_tree(
        X[right_mask],
        y[right_mask],
        max_depth,
        depth=depth + 1,
        min_samples_split=min_samples_split,
    )

    return node


# ==================== 7. 單筆預測：沿節點走到葉節點 ====================

def predict_one(point, tree):
    node = tree

    while not node["is_leaf"]:
        feature_index = node["feature_index"]
        threshold = node["threshold"]

        if point[feature_index] <= threshold:
            node = node["left"]
        else:
            node = node["right"]

    return node["prediction"]


# ==================== 8. 批次預測：逐筆呼叫 predict_one ====================

def predict(X_new, tree):
    predictions = []
    for point in X_new:
        prediction = predict_one(point, tree) # 呼叫單筆預測函式
        predictions.append(prediction)

    return np.array(predictions, dtype=np.int64)


# ==================== 9. 主線實驗 ====================

def main():
    rng_data = np.random.default_rng(CONFIG["data_seed"])

    X = rng_data.uniform(
        -2,
        2,
        size=(CONFIG["n_samples"], 2),
    )

    y = (
        X[:, 0] ** 2 + X[:, 1] ** 2 > 1.44
    ).astype(np.int64)

    print("資料形狀：", X.shape, y.shape)
    print("各類別數量：", np.bincount(y, minlength=2))

    # ==================== 切分資料 ====================

    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(len(X))

    n_train = int(len(X) * CONFIG["train_ratio"])
    n_validation = int(len(X) * CONFIG["validation_ratio"])

    # 先分編號，再用相同編號取出 X、y，保持樣本與標籤對應。
    train_indices = indices[:n_train]
    validation_indices = indices[n_train:n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_validation, y_validation = X[validation_indices], y[validation_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print("Train：", X_train.shape, y_train.shape)
    print("Validation：", X_validation.shape, y_validation.shape)
    print("Test：", X_test.shape, y_test.shape)

    # ==================== 比較深度：只使用 Train 與 Validation ====================

    results = {}

    for max_depth in CONFIG["max_depths"]:
        tree = build_tree(
            X_train,
            y_train,
            max_depth=max_depth,
            min_samples_split=CONFIG["min_samples_split"],
        )

        train_predictions = predict(X_train, tree)
        validation_predictions = predict(X_validation, tree)

        train_accuracy = np.mean(train_predictions == y_train)
        validation_accuracy = np.mean(validation_predictions == y_validation)

        results[max_depth] = {
            "tree": tree,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        }

        print(
            f"Depth={max_depth}："
            f"Train ACC={train_accuracy:.4f}，"
            f"Validation ACC={validation_accuracy:.4f}"
        )
    # Accuracy 越高越好；同分時保留候選順序中較早的深度。
    def get_validation_accuracy(depth):
        return results[depth]["validation_accuracy"]


    selected_depth = max(
        results,
        key=get_validation_accuracy,
    )

    selected_result = results[selected_depth]
    selected_tree = selected_result["tree"]

    print("選定最大深度：", selected_depth)
    print(
        "選定 Validation Accuracy：",
        selected_result["validation_accuracy"],
    )

    # Test 僅作最終評估，不用它重新選擇深度。
    # ==================== 最終 Test 評估 ====================

    test_predictions = predict(X_test, selected_tree)
    test_accuracy = np.mean(test_predictions == y_test)

    print("最終最大深度：", selected_depth)
    print("Test Accuracy：", test_accuracy)

    assert test_predictions.shape == y_test.shape


    # ==================== Majority-Class Baseline ====================

    train_counts = np.bincount(y_train, minlength=2)
    majority_class = np.argmax(train_counts)

    baseline_validation_predictions = np.full(
        y_validation.shape,
        majority_class,
        dtype=np.int64,
    )

    baseline_validation_accuracy = np.mean(
        baseline_validation_predictions == y_validation
    )

    print("Train 各類別數量：", train_counts)
    print("Baseline 固定預測類別：", majority_class)
    print("Baseline Validation Accuracy：", baseline_validation_accuracy)


    # ==================== 準備決策邊界網格 ====================

    grid_axis = np.linspace(-2, 2, 150)

    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)

    grid_points = np.column_stack((
        grid_x.ravel(),
        grid_y.ravel(),
    ))

    grid_predictions = predict(grid_points, selected_tree)
    grid_labels = grid_predictions.reshape(grid_x.shape)

    print("查詢點形狀：", grid_points.shape)
    print("網格標籤形狀：", grid_labels.shape)

    assert grid_predictions.shape == (len(grid_points),)

    # ==================== 畫出選定模型的決策區域 ====================

    fig, ax = plt.subplots(figsize=(7, 7))

    # ① 將網格的類別標籤轉成背景顏色。
    ax.contourf(
        grid_x,
        grid_y,
        grid_labels,
        levels=[-0.5, 0.5, 1.5],
        colors=["tab:blue", "tab:orange"],
        alpha=0.25,
    )

    # ② 疊上訓練資料，依真實標籤分色。
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

    # ③ 畫出資料生成規則的圓形邊界。
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
    ax.set_title(f"Decision Tree (max_depth={selected_depth})")
    ax.legend()

    fig.tight_layout()
    output_path = (
        Path(__file__).resolve().parent
        / "decision_tree_boundary.png"
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print("決策邊界圖已儲存：", output_path)

    # ==================== 比較樹深度與準確率 ====================

    depths = list(CONFIG["max_depths"])

    train_accuracies = [
        results[depth]["train_accuracy"]
        for depth in depths
    ]

    validation_accuracies = [
        results[depth]["validation_accuracy"]
        for depth in depths
    ]

    fig_scores, ax_scores = plt.subplots(figsize=(8, 5))

    ax_scores.plot(
        depths,
        train_accuracies,
        marker="o",
        label="Training Accuracy",
    )

    ax_scores.plot(
        depths,
        validation_accuracies,
        marker="o",
        label="Validation Accuracy",
    )

    ax_scores.axhline(
        baseline_validation_accuracy,
        color="gray",
        linestyle="--",
        label="Validation Baseline",
    )

    ax_scores.set_xticks(depths)
    ax_scores.set_ylim(0, 1.05)
    ax_scores.set_xlabel("Maximum Tree Depth")
    ax_scores.set_ylabel("Accuracy")
    ax_scores.set_title("Decision Tree Depth Comparison")
    ax_scores.legend()

    fig_scores.tight_layout()

    output_path = (
        Path(__file__).resolve().parent
        / "decision_tree_depth_comparison.png"
    )

    fig_scores.savefig(output_path, dpi=150, bbox_inches="tight")
    print("深度比較圖已儲存：", output_path)


    plt.close("all")


if __name__ == "__main__":
    main()
