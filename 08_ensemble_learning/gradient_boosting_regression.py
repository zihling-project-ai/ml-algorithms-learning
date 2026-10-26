"""L06｜平方誤差下的簡化 Gradient Boosting 回歸。
主線：固定切分 → 殘差學習與逐輪修正 → 比較輪數 → Validation 選模 → Test。
每棵基模型為一維回歸單層樹；圖表集中在 save_regression_plots。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==================== 1. 實驗設定 ====================

CONFIG = {
    "data_seed": 42,
    "split_seed": 42,
    "n_samples": 200,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "regression_noise_std": 0.5,
    "iterations_values": (10, 50, 100),
    "learning_rate": 0.1,
}

# ==================== 2. 回歸單層樹 ====================

def fit_regression_stump(x, targets):
    values = np.unique(x)
    thresholds = (values[:-1] + values[1:]) / 2

    # 沒有切點時，用整體平均值作為固定修正量。
    target_mean = np.mean(targets)
    best_stump = {
        "threshold": None,
        "left_value": target_mean,
        "right_value": target_mean,
    }
    best_score = np.inf

    for threshold in thresholds:
        left_mask = x <= threshold
        right_mask = ~left_mask

        left_targets = targets[left_mask]
        right_targets = targets[right_mask]

        left_value = np.mean(left_targets)
        right_value = np.mean(right_targets)

        # 以平方誤差總和評分，不使用分類的 Gini。
        score = (
            np.sum((left_targets - left_value) ** 2)
            + np.sum((right_targets - right_value) ** 2)
        )

        if score < best_score:
            best_score = score
            best_stump = {
                "threshold": threshold,
                "left_value": left_value,
                "right_value": right_value,
            }

    return best_stump


def predict_regression_stump(x, stump):
    if stump["threshold"] is None:
        return np.full(x.shape, stump["left_value"])

    return np.where(
        x <= stump["threshold"],
        stump["left_value"],
        stump["right_value"],
    )


# ==================== 3. Gradient Boosting：逐輪修正 ====================

def fit_gradient_boosting(x, y, learning_rate, iterations):
    # 起始預測只使用 Train 答案的平均值。
    initial_prediction = np.mean(y)
    current_predictions = np.full(y.shape, initial_prediction)

    stumps = []
    loss_history = [
        np.mean((y - current_predictions) ** 2)
    ]

    for _ in range(iterations):
        # ① 根據目前整體預測，重新計算殘差。
        residuals = y - current_predictions

        # ② 新模型學習這一輪尚未修正的部分。
        stump = fit_regression_stump(x, residuals)
        corrections = predict_regression_stump(x, stump)

        # ③ 加入本輪修正，成為下一輪的目前預測。
        current_predictions += learning_rate * corrections

        stumps.append(stump)
        loss_history.append(
            np.mean((y - current_predictions) ** 2)
        )

    return initial_prediction, stumps, np.array(loss_history)

# ==================== 4. Gradient Boosting：整合修正進行預測 ====================

def predict_gradient_boosting(
    x,
    initial_prediction,
    stumps,
    learning_rate,
):
    predictions = np.full(x.shape, initial_prediction)

    for stump in stumps:
        corrections = predict_regression_stump(x, stump)
        predictions += learning_rate * corrections

    return predictions

# ==================== 5. 圖表：與演算法主線分開 ====================

def save_regression_plots(
    x_train, y_train, x_validation, y_validation,
    selected_result, selected_iterations, results, learning_rate,
):
    output_folder = Path(__file__).resolve().parent
    grid_x = np.linspace(-2, 2, 500)
    grid_predictions = predict_gradient_boosting(
        grid_x, selected_result["initial_prediction"],
        selected_result["stumps"], learning_rate,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x_train, y_train, alpha=0.6, label="Training data")
    ax.scatter(x_validation, y_validation, marker="x", label="Validation data")
    ax.plot(grid_x, 3 * grid_x + 2, "k--", label="Generating rule: y = 3x + 2")
    ax.axhline(selected_result["initial_prediction"], color="gray",
               linestyle=":", label="Initial prediction: Train mean")
    ax.plot(grid_x, grid_predictions, color="tab:red",
            label=f"Gradient Boosting ({selected_iterations} stumps)")
    ax.set(xlabel="Feature x", ylabel="Target y",
           title="Gradient Boosting Regression Fit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fit_path = output_folder / "gradient_boosting_regression_fit.png"
    fig.savefig(fit_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 只展示已完成的訓練路徑，不增加候選設定或使用 Test。
    train_path = selected_result["loss_history"]
    validation_predictions = np.full(
        y_validation.shape, selected_result["initial_prediction"],
    )
    validation_path = [np.mean((y_validation - validation_predictions) ** 2)]
    for stump in selected_result["stumps"]:
        validation_predictions += learning_rate * predict_regression_stump(
            x_validation, stump,
        )
        validation_path.append(np.mean((y_validation - validation_predictions) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rounds = np.arange(len(train_path))
    axes[0].plot(rounds, train_path, label="Training MSE")
    axes[0].plot(rounds, validation_path, label="Validation MSE")
    axes[0].set(xlabel="Boosting round (0 = initial prediction)",
                ylabel="MSE", title="Selected Model: Sequential Correction")

    candidates = list(results)
    axes[1].plot(candidates, [results[n]["train_mse"] for n in candidates],
                 marker="o", label="Training MSE")
    axes[1].plot(candidates, [results[n]["validation_mse"] for n in candidates],
                 marker="o", label="Validation MSE")
    axes[1].set(xlabel="Number of stumps", ylabel="MSE",
                title="Candidate Iteration Comparison", xticks=candidates)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    loss_path = output_folder / "gradient_boosting_loss_comparison.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("擬合圖已儲存：", fit_path)
    print("Loss 比較圖已儲存：", loss_path)


# ==================== 6. 主線實驗 ====================

def main():
    # ==================== Gradient Boosting：回歸實驗起點 ====================

    rng_regression = np.random.default_rng(CONFIG["data_seed"])

    x_regression = rng_regression.uniform(
        -2,
        2,
        size=CONFIG["n_samples"],
    )

    noise = rng_regression.normal(
        0,
        CONFIG["regression_noise_std"],
        size=CONFIG["n_samples"],
    )

    y_regression = 3 * x_regression + 2 + noise

    # 固定切分；不依賴分類程式執行時建立的變數。
    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(len(x_regression))
    n_train = int(len(x_regression) * CONFIG["train_ratio"])
    n_validation = int(len(x_regression) * CONFIG["validation_ratio"])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train:n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    # 相同編號同時取出 Feature 與答案。
    x_reg_train = x_regression[train_indices]
    y_reg_train = y_regression[train_indices]
    x_reg_validation = x_regression[validation_indices]
    y_reg_validation = y_regression[validation_indices]
    x_reg_test = x_regression[test_indices]
    y_reg_test = y_regression[test_indices]

    print("Train／Validation／Test 筆數：",
          len(y_reg_train), len(y_reg_validation), len(y_reg_test))

    # ==================== 比較修正輪數 ====================

    results = {}

    for iterations in CONFIG["iterations_values"]:
        initial_prediction, reg_stumps, loss_history = fit_gradient_boosting(
            x_reg_train,
            y_reg_train,
            learning_rate=CONFIG["learning_rate"],
            iterations=iterations,
        )

        validation_predictions = predict_gradient_boosting(
            x_reg_validation,
            initial_prediction,
            reg_stumps,
            learning_rate=CONFIG["learning_rate"],
        )

        assert len(reg_stumps) == iterations
        assert len(loss_history) == iterations + 1
        assert validation_predictions.shape == y_reg_validation.shape

        train_mse = loss_history[-1]
        validation_mse = np.mean(
            (y_reg_validation - validation_predictions) ** 2
        )

        results[iterations] = {
            "initial_prediction": initial_prediction,
            "stumps": reg_stumps,
            "loss_history": loss_history,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
        }

        print(
            f"Iterations={iterations}："
            f"Train MSE={train_mse:.4f}，"
            f"Validation MSE={validation_mse:.4f}"
        )

    # ==================== 依 Validation 選擇輪數 ====================

    def get_validation_mse(iterations):
        return results[iterations]["validation_mse"]


    selected_iterations = min(
        results,
        key=get_validation_mse,
    )

    selected_result = results[selected_iterations]

    print("選定修正輪數：", selected_iterations)
    print(
        "選定 Validation MSE：",
        selected_result["validation_mse"],
    )

    # ==================== 最終 Test 評估 ====================
    # 模型已選定；不根據 Test 調參或重選輪數。

    test_predictions = predict_gradient_boosting(
        x_reg_test,
        selected_result["initial_prediction"],
        selected_result["stumps"],
        learning_rate=CONFIG["learning_rate"],
    )

    test_mse = np.mean(
        (y_reg_test - test_predictions) ** 2
    )

    print("Test MSE：", test_mse)

    assert test_predictions.shape == y_reg_test.shape

    print("起始 Training MSE：", selected_result["loss_history"][0])
    save_regression_plots(
        x_reg_train, y_reg_train, x_reg_validation, y_reg_validation,
        selected_result, selected_iterations, results, CONFIG["learning_rate"],
    )

if __name__ == "__main__":
    main()
