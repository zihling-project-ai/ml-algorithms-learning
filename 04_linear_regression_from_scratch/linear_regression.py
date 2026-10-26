"""主線：合成資料 → 固定切分 → 比較學習率 → Validation 選擇 → Test → 存圖。

輔助練習另見 linear_regression_learning.py。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 1. 實驗設定 ====================

CONFIG = {
    "initial_weight": 1.0,
    "initial_bias": 2.5,
    "iterations": 200,
    "data_seed": 42,
    "n_samples": 200,
    "noise_std": 0.5,
    "split_seed": 42,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "learning_rates": (0.01, 0.1, 0.5),
}


# ==================== 2. 預測與 MSE ====================

def predict(x, weight, bias):
    return weight * x + bias


def compute_mse(y_true, y_pred):
    errors = y_pred - y_true
    return np.mean(errors ** 2)


# ==================== 3. 解析梯度 ====================

def compute_gradients(x, y_true, y_pred):
    """解析梯度：直接計算 MSE 對兩個參數的變化率。"""
    errors = y_pred - y_true

    grad_weight = 2 * np.mean(errors * x)
    grad_bias = 2 * np.mean(errors)

    return grad_weight, grad_bias


# ==================== 4. 訓練函式 ====================

def fit_linear_regression(
    x, y_true, initial_weight, initial_bias, learning_rate, iterations
):
    weight = initial_weight
    bias = initial_bias

    # 先記錄更新前的 MSE，作為第 0 輪。
    initial_predictions = predict(x, weight, bias)
    loss_history = [compute_mse(y_true, initial_predictions)]

    for _ in range(iterations):
        predictions = predict(x, weight, bias)

        # 兩個梯度都用同一組目前參數計算。
        grad_weight, grad_bias = compute_gradients(
            x, y_true, predictions
        )

        # 往梯度反方向更新。
        weight = weight - learning_rate * grad_weight
        bias = bias - learning_rate * grad_bias

        updated_predictions = predict(x, weight, bias)
        loss = compute_mse(y_true, updated_predictions)
        loss_history.append(loss)

    # 在迴圈外回傳，完成所有更新後才結束函式。
    return weight, bias, np.array(loss_history)



def main():
    # ==================== 5. 建立 200 筆帶雜訊資料 ====================

    rng = np.random.default_rng(CONFIG["data_seed"])

    X = rng.uniform(-2, 2, size=(CONFIG["n_samples"], 1))
    x = X[:, 0]

    noise = rng.normal(
        0,
        CONFIG["noise_std"],
        size=CONFIG["n_samples"],
    )

    y = 3 * x + 2 + noise


    # ==================== 切分 Train / Validation / Test ====================

    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(len(X))

    n_train = int(len(x) * CONFIG["train_ratio"])
    n_validation = int(len(x) * CONFIG["validation_ratio"])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train: n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    x_train, y_train = x[train_indices], y[train_indices]
    x_validation, y_validation = x[validation_indices], y[validation_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    print("Train：", x_train.shape, y_train.shape)
    print("Validation：", x_validation.shape, y_validation.shape)
    print("Test：", x_test.shape, y_test.shape)


    # ==================== 6. 使用解析梯度訓練 ====================

    result = {}

    for learning_rate in CONFIG["learning_rates"]:
        weight, bias, loss_history = fit_linear_regression(
            x_train,
            y_train,
            initial_weight=CONFIG["initial_weight"],
            initial_bias=CONFIG["initial_bias"],
            learning_rate=learning_rate,
            iterations=CONFIG["iterations"],
        )

        train_predictions = predict(x_train, weight, bias)
        validation_predictions = predict(x_validation, weight, bias)
        train_mse = compute_mse(y_train, train_predictions)
        validation_mse = compute_mse(y_validation, validation_predictions)

        result[learning_rate] = {
            "weight": weight,
            "bias":bias,
            "loss_history":loss_history,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
        }

        print(
            f"Learning Rate={learning_rate}",
            f"Train MSE={train_mse:.6f}",
            f"Validation MSE={validation_mse:.6f}"
        )





    # ==================== 最終 Test 評估 ====================

    def get_validation_mse(rate):
        return result[rate]["validation_mse"]

    selected_learning_rate = min(result, key=get_validation_mse)
    selected_result = result[selected_learning_rate]
    print("Validation MSE：", selected_result["validation_mse"])

    selected_weight = selected_result["weight"]
    selected_bias = selected_result["bias"]

    test_predictions = predict(x_test, selected_weight, selected_bias)
    test_mse = compute_mse(y_test, test_predictions)

    print("選定 Learning Rate：", selected_learning_rate)
    print("最終 Weight：", selected_weight)
    print("最終 Bias：", selected_bias)
    print("Test MSE：", test_mse)

    # ==================== 保存實驗圖表 ====================

    output_dir = Path(__file__).resolve().parent


    # ---------- 1. 選定模型的擬合直線 ----------

    fig_fit, ax = plt.subplots()

    ax.scatter(
        x_train, y_train,
        alpha=0.5,
        label="Train",
    )
    ax.scatter(
        x_validation, y_validation,
        marker="x",
        alpha=0.7,
        label="Validation",
    )

    line_x = np.linspace(-2, 2, 100)

    ax.plot(
        line_x,
        3 * line_x + 2,
        color="black",
        linestyle="--",
        label="Generating rule",
    )

    ax.plot(
        line_x,
        predict(line_x, selected_weight, selected_bias),
        color="tab:red",
        label="Selected model",
    )

    ax.set_xlabel("Feature x")
    ax.set_ylabel("Target y")
    ax.set_title(
        f"Linear Regression Fit (LR={selected_learning_rate})"
    )
    ax.legend()

    fig_fit.tight_layout()

    fit_path = output_dir / "linear_regression_selected_fit.png"
    fig_fit.savefig(fit_path, dpi=150, bbox_inches="tight")
    plt.close(fig_fit)


    # ---------- 2. 三種學習率的 Training Loss ----------

    fig_loss, ax = plt.subplots()

    for learning_rate in CONFIG["learning_rates"]:
        history = result[learning_rate]["loss_history"]
        rounds = np.arange(len(history))

        ax.plot(
            rounds,
            history,
            label=f"Learning rate={learning_rate}",
        )

    ax.set_xlabel("Updates")
    ax.set_ylabel("Training MSE")
    ax.set_title("Learning Rate Comparison")
    ax.legend()

    fig_loss.tight_layout()

    loss_path = output_dir / "linear_regression_learning_rates.png"
    fig_loss.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig_loss)

    print("擬合圖已儲存：", fit_path)
    print("Loss 比較圖已儲存：", loss_path)


if __name__ == "__main__":
    main()
