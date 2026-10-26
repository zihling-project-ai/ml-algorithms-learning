"""輔助學習與驗證：保留早期單層樹、逐步範例及已完成的測試。
不加入新演算法；執行本檔才會跑檢查，匯入主線不會執行示範。
"""
import numpy as np
from decision_tree import (
    compute_gini, find_best_threshold, find_best_split,
    make_leaf, prepare_node, build_tree, predict_one, predict,
)

# ==================== A. 早期學習版本：不在多層建樹流程中使用 ====================

def fit_stump(X, y):
    """
    ① 找出切分規則
    ② 按規則分左右
    ③ 左右各選一個多數類別
    ④ 保存規則與答案
    """
    split = find_best_split(X, y)

    if split is None:
        raise ValueError("沒有可用的切分")

    feature_index = split["feature_index"]
    threshold = split["threshold"]

    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    y_left = y[left_mask]
    y_right = y[right_mask]

    # bincount 數票，argmax 找出最多票的類別編號。
    left_prediction = np.argmax(np.bincount(y_left))
    right_prediction = np.argmax(np.bincount(y_right))

    return {
        "feature_index": feature_index,
        "threshold": threshold,
        "left_prediction": left_prediction,
        "right_prediction": right_prediction,
    }


def predict_stump(X_new, stump):
    feature_index = stump["feature_index"]
    threshold = stump["threshold"]

    predictions = []

    for point in X_new:
        # point 是單筆樣本，取模型指定的 Feature。
        feature_value = point[feature_index]

        if feature_value <= threshold:
            prediction = stump["left_prediction"]
        else:
            prediction = stump["right_prediction"]

        predictions.append(prediction)

    # 完成全部樣本後才回傳。
    return np.array(predictions, dtype=np.int64)


def build_one_level_tree(X, y):
    node = prepare_node(X, y)

    # 已經不需要切，直接交回葉節點。
    if node["is_leaf"]:
        return node

    feature_index = node["feature_index"]
    threshold = node["threshold"]

    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    # 左右各保存一個葉節點。
    node["left"] = make_leaf(y[left_mask])
    node["right"] = make_leaf(y[right_mask])

    return node


def predict_one_level(point, tree):
    if tree["is_leaf"]:
        return tree["prediction"]
    feature_value = point[tree["feature_index"]]

    if feature_value <= tree["threshold"]:
        return tree["left"]["prediction"]   # 左葉節點答案
    else:
        return tree["right"]["prediction"]  # 右葉節點答案


# ==================== B. 已完成的範例與驗證 ====================

def run_checks():
    # ==================== 6. 基本 Gini 檢查 ====================

    assert np.isclose(compute_gini(np.array([0, 0])), 0.0)
    assert np.isclose(compute_gini(np.array([0, 0, 1, 1])), 0.5)


    # ==================== 7. 單欄搜尋與改善量 ====================

    # 小寫 x：一維，只有單一 Feature 的數值。
    x_demo = np.array([1.0, 2.0, 3.0, 4.0])
    y_demo = np.array([0, 0, 1, 1])

    best_threshold, best_score = find_best_threshold(x_demo, y_demo)

    parent_gini = compute_gini(y_demo)
    gini_decrease = parent_gini - best_score

    print("單欄最佳 Threshold：", best_threshold)
    print("切分前 Gini：", parent_gini)
    print("切分後加權 Gini：", best_score)
    print("Gini 改善量：", gini_decrease)

    assert np.isclose(best_threshold, 2.5)
    assert np.isclose(best_score, 0.0)
    assert np.isclose(gini_decrease, 0.5)


    no_gain_tree = build_tree(
    np.array([[1.0], [1.0], [2.0], [2.0]]),
    np.array([0, 1, 0, 1]),
    max_depth=3,
    )

    print("沒有改善的節點：", no_gain_tree)

    assert no_gain_tree["is_leaf"]
    assert no_gain_tree["prediction"] == 0


    # ==================== 8. 多欄資料建立單層樹 ====================

    # 大寫 X：二維，每列一筆樣本，每欄一個 Feature。
    X_demo = np.array([
        [1.0, 10.0],
        [2.0, 30.0],
        [3.0, 20.0],
        [4.0, 40.0],
    ])

    stump = fit_stump(X_demo, y_demo)

    print("\n單層樹：", stump)

    assert stump["feature_index"] == 0
    assert np.isclose(stump["threshold"], 2.5)
    assert stump["left_prediction"] == 0
    assert stump["right_prediction"] == 1


    # ==================== 9. 預測新資料 ====================

    X_new = np.array([
        [2.0, 99.0],  # 第一欄小於 2.5 → 左邊
        [3.0, -5.0],  # 第一欄大於 2.5 → 右邊
        [2.5, 10.0],  # 等於 Threshold → 左邊
    ])

    predictions = predict_stump(X_new, stump)

    print("新資料預測：", predictions)
    assert np.array_equal(predictions, np.array([0, 1, 0]))


    # ==================== 10. 已純群組的限制觀察 ====================

    y_pure = np.array([0, 0, 0, 0])

    pure_threshold, pure_score = find_best_threshold(x_demo, y_pure)
    pure_gain = compute_gini(y_pure) - pure_score

    print("\n純群組的最佳候選：", pure_threshold)
    print("純群組的 Gini 改善量：", pure_gain)

    assert np.isclose(pure_gain, 0.0)

    # 單獨找候選可能回傳零改善切點；prepare_node 已處理純群停止。
    # prepare_node 也會在混合標籤的最佳改善量不超過 1e-12 時停止。

    print("\n目前單層樹檢查全部通過")


    # A. 標籤已純，即使 Feature 能切，也不需要切。
    pure_node = prepare_node(
        np.array([[1.0], [2.0], [3.0]]),
        np.array([1, 1, 1]),
    )

    # B. 標籤混合，但所有 Feature 相同，無法切。
    constant_node = prepare_node(
        np.array([[2.0], [2.0], [2.0]]),
        np.array([0, 1, 1]),
    )

    print("純群組節點：", pure_node)
    print("同值 Feature 節點：", constant_node)

    assert pure_node["is_leaf"]
    assert pure_node["prediction"] == 1
    assert constant_node["is_leaf"]
    assert constant_node["prediction"] == 1


    X_demo = np.array([
        [1.0, 10.0],
        [2.0, 30.0],
        [3.0, 20.0],
        [4.0, 40.0],
    ])
    y_demo = np.array([0, 0, 1, 1])

    tree = build_one_level_tree(X_demo, y_demo)

    print("根節點欄位：", tree["feature_index"])
    print("根節點切點：", tree["threshold"])
    print("左邊答案：", tree["left"]["prediction"])
    print("右邊答案：", tree["right"]["prediction"])

    assert not tree["is_leaf"]
    assert tree["left"]["is_leaf"]
    assert tree["right"]["is_leaf"]
    assert tree["left"]["prediction"] == 0
    assert tree["right"]["prediction"] == 1


    assert predict_one_level(np.array([2.0, 99.0]), tree) == 0
    assert predict_one_level(np.array([3.0, -5.0]), tree) == 1
    assert predict_one_level(np.array([2.5, 10.0]), tree) == 0

    # 根節點本身是葉節點時，也應能直接作答。
    leaf = make_leaf(np.array([1, 1, 0]))
    assert predict_one_level(np.array([100.0, 200.0]), leaf) == 1

    print("一層樹預測測試通過")


    X_tree_demo = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_tree_demo = np.array([0, 1, 1, 0])

    batch_test_tree = build_tree(
        X_tree_demo,
        y_tree_demo,
        max_depth=2,
    )

    X_batch = np.array([[1.0], [2.0], [4.0]])

    batch_predictions = predict(X_batch, batch_test_tree)

    print("整批預測：", batch_predictions)
    assert np.array_equal(batch_predictions, np.array([0, 1, 0]))

    small_tree = build_tree(
        np.array([[1.0], [2.0], [3.0]]),
        np.array([0, 1, 1]),
        max_depth=3,
        min_samples_split=4,
    )

    print("筆數不足的節點：", small_tree)

    assert small_tree["is_leaf"]
    assert small_tree["prediction"] == 1


    boundary_tree = build_tree(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([0, 0, 1, 1]),
        max_depth=2,
        min_samples_split=4,
    )

    assert not boundary_tree["is_leaf"]
    assert np.isclose(boundary_tree["threshold"], 2.5)

    print("剛好達到筆數門檻，可以切分")

    print("\n全部既有檢查通過")


if __name__ == "__main__":
    run_checks()
