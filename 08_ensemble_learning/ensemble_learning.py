"""L06：比較 Single Tree、Bagging、Random Forest 與 AdaBoost。"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


# ==================== 1. 沿用既有功能 ====================

forest_folder = (
    Path(__file__).resolve().parent.parent
    / "07_random_forest_from_scratch"
)
sys.path.insert(0, str(forest_folder))

from random_forest import bootstrap_sampling, predict_random_forest, fit_random_forest
from decision_tree import find_best_threshold, make_leaf, predict


# ==================== 2. 實驗設定 ====================

CONFIG = {
    "data_seed": 42,
    "split_seed": 42,
    "bagging_seed": 42,
    "n_samples": 200,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "max_depth": 10,
    "min_samples_split": 2,
    "n_estimators": 10,
    "forest_seed": 42,
    "max_features": 1,
    "boosting_rounds": 10,
    "boosting_max_depth": 1,
}


# ==================== 3. 搜尋全部 Features 的最佳切分 ====================

def find_best_split(X, y):
    best_split = None
    best_score = np.inf

    # 固定依第 0、1……欄搜尋；同分保留先找到的規則。
    for feature_index in range(X.shape[1]):
        threshold, score = find_best_threshold(
            X[:, feature_index],
            y,
        )

        if threshold is None:
            continue

        if score < best_score:
            best_score = score
            best_split = {
                "feature_index": feature_index,
                "threshold": threshold,
            }

    return best_split


# ==================== 4. Single Tree 與 Bagging 共用建樹 ====================

def build_tree(
    X,
    y,
    max_depth,
    min_samples_split,
    depth=0,
):
    # ① 到達限制，或標籤已純，就保存多數類別。
    if (
        depth >= max_depth
        or len(y) < min_samples_split
        or len(np.unique(y)) == 1
    ):
        return make_leaf(y)

    # ② 搜尋全部欄位，選最低加權 Gini 的切分。
    split = find_best_split(X, y)

    if split is None:
        return make_leaf(y)

    # ③ 用同一組 mask 分出對應的 Features 與標籤。
    feature_index = split["feature_index"]
    threshold = split["threshold"]

    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    node = {
        "is_leaf": False,
        "feature_index": feature_index,
        "threshold": threshold,
    }

    # ④ 左右子集使用相同流程，各自繼續建樹。
    node["left"] = build_tree(
        X[left_mask],
        y[left_mask],
        max_depth,
        min_samples_split,
        depth=depth + 1,
    )

    node["right"] = build_tree(
        X[right_mask],
        y[right_mask],
        max_depth,
        min_samples_split,
        depth=depth + 1,
    )

    return node


# ==================== Bagging：抽樣並訓練多棵樹 ====================

def fit_bagging(
    X,
    y,
    n_estimators,
    max_depth,
    min_samples_split,
    seed,
):
    rng = np.random.default_rng(seed)
    trees = []

    for _ in range(n_estimators):
        # 每棵樹取得不同的 Bootstrap 樣本組合。
        X_sample, y_sample = bootstrap_sampling(X, y, rng)

        # 使用與 Single Tree 完全相同的建樹函式。
        tree = build_tree(
            X_sample,
            y_sample,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

        trees.append(tree)

    return trees


# ==================== AdaBoost：加權 Gini ====================

def compute_weighted_gini(y, sample_weights):
    # 分別加總各類別的權重。
    class_weights = np.bincount(
        y,
        weights=sample_weights,
        minlength=2,
    )

    # 節點內的權重不一定加總為 1，因此要重新換成比例。
    proportions = class_weights / np.sum(sample_weights)

    return 1 - np.sum(proportions ** 2)

# ==================== AdaBoost：訓練加權單層樹 ====================

def fit_weighted_stump(X, y, sample_weights):
    """搜尋加權 Gini 最低的切分，左右葉節點依權重投票。"""
    best_split = None
    best_score = np.inf
    total_weight = np.sum(sample_weights)

    # ① 逐欄、逐切點比較；改用權重衡量左右的重要性。
    for feature_index in range(X.shape[1]):
        x_column = X[:, feature_index]
        values = np.unique(x_column)
        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left_mask = x_column <= threshold
            right_mask = ~left_mask

            left_weights = sample_weights[left_mask]
            right_weights = sample_weights[right_mask]
            left_weight = np.sum(left_weights)
            right_weight = np.sum(right_weights)

            if left_weight <= 0 or right_weight <= 0:
                continue

            left_gini = compute_weighted_gini(y[left_mask], left_weights)
            right_gini = compute_weighted_gini(y[right_mask], right_weights)
            score = (
                left_weight / total_weight * left_gini
                + right_weight / total_weight * right_gini
            )

            if score < best_score:
                best_score = score
                best_split = {
                    "feature_index": feature_index,
                    "threshold": threshold,
                }

    # ② 葉節點不數人頭，而是加總各類別的樣本權重。
    def weighted_leaf(labels, weights):
        class_weights = np.bincount(
            labels, weights=weights, minlength=2,
        )
        return {
            "is_leaf": True,
            "prediction": np.argmax(class_weights),
        }

    if best_split is None:
        return weighted_leaf(y, sample_weights)

    # ③ 依最佳規則分左右，保存葉節點答案；沿用 predict 即可預測。
    feature_index = best_split["feature_index"]
    threshold = best_split["threshold"]
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    return {
        "is_leaf": False,
        "feature_index": feature_index,
        "threshold": threshold,
        "left": weighted_leaf(y[left_mask], sample_weights[left_mask]),
        "right": weighted_leaf(y[right_mask], sample_weights[right_mask]),
    }


# ==================== AdaBoost：逐輪訓練 ====================

def fit_adaboost(X, y, rounds):
    if len(y) == 0 or len(X) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("本版 AdaBoost 需要非空、配對的 X 與 0/1 標籤")
    if isinstance(rounds, (bool, np.bool_)) or not isinstance(rounds, (int, np.integer)) or rounds < 1:
        raise ValueError("rounds 必須是正整數")
    sample_weights = np.full(len(y), 1.0 / len(y))

    stumps = []
    alphas = []
    error_history = []

    for iteration in range(rounds):
        stump = fit_weighted_stump(X, y, sample_weights)
        predictions = predict(X, stump)

        incorrect = predictions != y
        weighted_error = np.sum(sample_weights[incorrect])

        # 沒有優於加權隨機二選一，就不加入這棵樹。
        if weighted_error >= 0.5:
            print("停止：本輪加權錯誤率未低於 0.5")
            break

        # 避免錯誤率為 0 時，公式出現除以零或 log(0)。
        safe_error = np.clip(weighted_error, 1e-12, 1 - 1e-12)
        alpha = 0.5 * np.log((1 - safe_error) / safe_error)

        stumps.append(stump)
        alphas.append(alpha)
        error_history.append(weighted_error)

        print(
            f"AdaBoost 第 {iteration + 1} 輪："
            f"加權錯誤率={weighted_error:.4f}，"
            f"alpha={alpha:.4f}"
        )

        # 完全答對時，保存本棵後停止。
        if weighted_error == 0:
            break

        # 更新重要性，交給下一輪使用。
        sample_weights[incorrect] *= np.exp(alpha)
        sample_weights[~incorrect] *= np.exp(-alpha)
        sample_weights /= np.sum(sample_weights)

    return stumps, np.array(alphas), np.array(error_history)


# ==================== AdaBoost：加權投票 ====================

def predict_adaboost(X_new, stumps, alphas):
    if len(stumps) != len(alphas):
        raise ValueError("每棵單層樹都必須有一個 alpha")
    if len(stumps) == 0:
        raise ValueError("沒有可供預測的 AdaBoost 模型")

    # 每列一筆樣本；兩欄分別保存類別 0、1 的票重。
    class_scores = np.zeros((len(X_new), 2))

    for stump, alpha in zip(stumps, alphas):
        predictions = predict(X_new, stump)

        class_scores[:, 0] += alpha * (predictions == 0)
        class_scores[:, 1] += alpha * (predictions == 1)

    # 每筆樣本選票重較大的類別；同票選 0。
    return np.argmax(class_scores, axis=1)


# ==================== 5. 主線實驗 ====================

def main():
    # 重現相同的圓形資料。
    rng_data = np.random.default_rng(CONFIG["data_seed"])

    X = rng_data.uniform(
        -2,
        2,
        size=(CONFIG["n_samples"], 2),
    )
    y = (
        X[:, 0] ** 2 + X[:, 1] ** 2 > 1.44
    ).astype(np.int64)

    # 固定切分，三種模型共用相同資料。
    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(len(X))

    n_train = int(len(X) * CONFIG["train_ratio"])
    n_validation = int(len(X) * CONFIG["validation_ratio"])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train:n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_validation, y_validation = (
        X[validation_indices],
        y[validation_indices],
    )
    X_test, y_test = X[test_indices], y[test_indices]

    # 先建立比較起點：使用全部 Train 的單棵樹。
    single_tree = build_tree(
        X_train,
        y_train,
        max_depth=CONFIG["max_depth"],
        min_samples_split=CONFIG["min_samples_split"],
    )

    train_predictions = predict(X_train, single_tree)
    validation_predictions = predict(X_validation, single_tree)

    print("Train／Validation／Test 筆數：",
          len(y_train), len(y_validation), len(y_test))
    print(
        "Single Tree Training Accuracy：",
        np.mean(train_predictions == y_train),
    )
    print(
        "Single Tree Validation Accuracy：",
        np.mean(validation_predictions == y_validation),
    )

    assert validation_predictions.shape == y_validation.shape

    # ==================== Bagging 訓練與評估 ====================

    bagging_trees = fit_bagging(
        X_train,
        y_train,
        n_estimators=CONFIG["n_estimators"],
        max_depth=CONFIG["max_depth"],
        min_samples_split=CONFIG["min_samples_split"],
        seed=CONFIG["bagging_seed"],
    )

    # 沿用已完成的 Hard Voting；它也適用於 Bagging 的樹清單。
    bagging_train_predictions = predict_random_forest(
        X_train,
        bagging_trees,
    )

    bagging_validation_predictions = predict_random_forest(
        X_validation,
        bagging_trees,
    )

    print("Bagging 樹數：", len(bagging_trees))
    print(
        "Bagging Training Accuracy：",
        np.mean(bagging_train_predictions == y_train),
    )
    print(
        "Bagging Validation Accuracy：",
        np.mean(bagging_validation_predictions == y_validation),
    )

    assert len(bagging_trees) == CONFIG["n_estimators"]
    assert bagging_validation_predictions.shape == y_validation.shape


    # ==================== Random Forest 訓練與評估 ====================

    forest_trees = fit_random_forest(
        X_train,
        y_train,
        n_estimators=CONFIG["n_estimators"],
        max_depth=CONFIG["max_depth"],
        min_samples_split=CONFIG["min_samples_split"],
        max_features=CONFIG["max_features"],
        seed=CONFIG["forest_seed"],
    )

    forest_train_predictions = predict_random_forest(
        X_train,
        forest_trees,
    )

    forest_validation_predictions = predict_random_forest(
        X_validation,
        forest_trees,
    )

    print(
        "Random Forest Training Accuracy：",
        np.mean(forest_train_predictions == y_train),
    )
    print(
        "Random Forest Validation Accuracy：",
        np.mean(forest_validation_predictions == y_validation),
    )

    assert len(forest_trees) == CONFIG["n_estimators"]
    assert forest_validation_predictions.shape == y_validation.shape

    # ==================== Boosting：第一輪準備 ====================

    # 一開始，每筆 Training 樣本都一樣重要。
    sample_weights = np.full(
        len(y_train),
        1.0 / len(y_train),
    )

    # 換成新的加權單層樹訓練函式。
    first_stump = fit_weighted_stump(
        X_train,
        y_train,
        sample_weights,
    )


    stump_predictions = predict(X_train, first_stump)

    # 答錯的位置為 True。
    incorrect = stump_predictions != y_train

    # 將答錯樣本的權重加總。
    weighted_error = np.sum(sample_weights[incorrect])

    print("初始每筆樣本權重：", sample_weights[0])
    print("樣本權重總和：", np.sum(sample_weights))
    print("第一棵單層樹答錯筆數：", np.count_nonzero(incorrect))
    print("第一棵單層樹加權錯誤率：", weighted_error)

    assert np.isclose(np.sum(sample_weights), 1.0)

    weighted_gini = compute_weighted_gini(
        y_train,
        sample_weights,
    )

    print("初始 Training 加權 Gini：", weighted_gini)

    # ==================== AdaBoost：第一次權重更新 ====================

    # 本區塊先處理錯誤率介於 0 與 0.5 的情況。
    if not 0 < weighted_error < 0.5:
        raise ValueError("本輪示範需要 0 < weighted_error < 0.5")

    alpha = 0.5 * np.log(
        (1 - weighted_error) / weighted_error
    )

    # 答錯的乘上大於 1 的倍率；答對的乘上小於 1 的倍率。
    updated_weights = sample_weights.copy()

    updated_weights[incorrect] *= np.exp(alpha)
    updated_weights[~incorrect] *= np.exp(-alpha)

    # 重新調整比例，讓全部權重加總為 1。
    updated_weights = updated_weights / np.sum(updated_weights)

    print("第一棵樹的投票權重：", alpha)
    print("答錯樣本更新後的平均權重：",
        np.mean(updated_weights[incorrect]))
    print("答對樣本更新後的平均權重：",
        np.mean(updated_weights[~incorrect]))

    assert np.isclose(np.sum(updated_weights), 1.0)

    # ==================== 第二棵樹：使用更新後的權重 ====================

    second_stump = fit_weighted_stump(
        X_train,
        y_train,
        updated_weights,
    )

    second_predictions = predict(X_train, second_stump)
    second_incorrect = second_predictions != y_train

    # 兩棵樹都使用同一組新權重評估。
    first_error = np.sum(updated_weights[incorrect])
    second_error = np.sum(updated_weights[second_incorrect])

    print(
        "第一棵規則：",
        first_stump["feature_index"],
        first_stump["threshold"],
    )
    print(
        "第二棵規則：",
        second_stump["feature_index"],
        second_stump["threshold"],
    )
    print("新權重下，第一棵錯誤率：", first_error)
    print("新權重下，第二棵錯誤率：", second_error)
    #plt.show()

    # ==================== 訓練 AdaBoost ====================

    boost_stumps, boost_alphas, boost_errors = fit_adaboost(
        X_train,
        y_train,
        rounds=CONFIG["boosting_rounds"],
    )

    print("實際完成輪數：", len(boost_stumps))

    assert len(boost_stumps) == len(boost_alphas)


    boost_train_predictions = predict_adaboost(
    X_train, boost_stumps, boost_alphas,
    )

    boost_validation_predictions = predict_adaboost(
        X_validation, boost_stumps, boost_alphas,
    )

    print(
        "AdaBoost Training Accuracy：",
        np.mean(boost_train_predictions == y_train),
    )
    print(
        "AdaBoost Validation Accuracy：",
        np.mean(boost_validation_predictions == y_validation),
    )

    assert boost_validation_predictions.shape == y_validation.shape

    # ==================== 比較四種模型的決策邊界 ====================

    # 準備相同的網格，觀察四種模型。
    grid_axis = np.linspace(-2, 2, 150)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)

    grid_points = np.column_stack((
        grid_x.ravel(),
        grid_y.ravel(),
    ))

    single_grid_predictions = predict(grid_points, single_tree)
    bagging_grid_predictions = predict_random_forest(
        grid_points,
        bagging_trees,
    )

    forest_grid_predictions = predict_random_forest(
        grid_points,
        forest_trees,
    )

    boost_grid_predictions = predict_adaboost(
        grid_points, boost_stumps, boost_alphas,
    )

    # 每組保存：圖名、網格預測結果。
    plot_data = [
        ("Single Tree", single_grid_predictions),
        (f"Bagging ({CONFIG['n_estimators']} trees)", bagging_grid_predictions),
        (f"Random Forest ({CONFIG['n_estimators']} trees)", forest_grid_predictions),
        (f"AdaBoost ({len(boost_stumps)} stumps)", boost_grid_predictions),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 12),
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    for ax, (title, grid_predictions) in zip(axes, plot_data):
        grid_labels = grid_predictions.reshape(grid_x.shape)

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
                s=25,
                label=name,
            )

        ax.add_patch(plt.Circle(
            (0, 0),
            radius=1.2,
            fill=False,
            color="black",
            linestyle="--",
            label="True boundary",
        ))

        ax.set_title(title)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_xlabel("Feature 1")

    for ax in axes:
        ax.set_ylabel("Feature 2")
    axes[-1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    output_path = (
        Path(__file__).resolve().parent
        / "ensemble_classification_comparison.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print("比較圖已儲存：", output_path)

    #plt.show()

if __name__ == "__main__":
    main()
