"""主線：資料 → 切分 → 學習率比較 → Validation 選擇 → Test。
輔助小例子與有限差分另見 logistic_regression_learning.py。
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==================== 1. 實驗設定 ====================

CONFIG = {
    "data_seed": 42,
    "n_samples": 200,
    "split_seed": 42,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "initial_weight": 1.0,
    "initial_bias": 0.0,
    "threshold": 0.5,
    "iterations": 200,
    "learning_rates": (0.01, 0.1, 0.5),
}


# ==================== 2. 核心函式 ====================

def sigmoid(z):
    """把線性分數轉成類別 1 的機率。"""
    # exp 的輸入永遠非正，避免很大的負分數造成溢位。
    z = np.asarray(z, dtype=float)
    exp_neg_abs = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + exp_neg_abs), exp_neg_abs / (1 + exp_neg_abs))


def predict_proba(x, weight, bias):
    """先算線性分數，再轉成機率。"""
    z = weight * x + bias
    return sigmoid(z)


def predict_labels(p, threshold=0.5):
    """達到門檻預測 1，否則預測 0。"""
    return (p >= threshold).astype(np.int64)


def compute_bce(y_true, p):
    """衡量模型給真實類別的機率，越低越好。"""
    # 避免 log(0)；這不是分類門檻。
    safe_p = np.clip(p, 1e-12, 1 - 1e-12)

    # y=1：留下 -log(p)。
    # y=0：留下 -log(1-p)。
    term_1 = y_true * np.log(safe_p)
    term_0 = (1 - y_true) * np.log(1 - safe_p)
    losses = -(term_1 + term_0)

    return np.mean(losses)

def compute_gradients(x, y_true, p):
    errors = p - y_true

    grad_weight = np.mean(errors * x)
    grad_bias = np.mean(errors)

    return grad_weight, grad_bias

def fit_logistic_regression(x, y_true, initial_weight, initial_bias, learning_rate, iterations):
    weight = initial_weight
    bias = initial_bias

    initial_p = predict_proba(x, weight, bias)
    loss_history = [compute_bce(y_true, initial_p)]


    for _ in range(iterations):
        p = predict_proba(x, weight, bias)
        grad_weight, grad_bias = compute_gradients(x, y_true, p)

        new_weight = weight - learning_rate * grad_weight
        new_bias = bias - learning_rate *grad_bias

        updated_p = predict_proba(x, new_weight, new_bias)
        loss_history.append(compute_bce(y_true, updated_p))

        weight = new_weight
        bias = new_bias

    return weight, bias, np.array(loss_history)


def main():
    # ==================== 4. 主線：建立二元分類資料 ====================

    rng = np.random.default_rng(CONFIG["data_seed"])

    # 200 列：200 筆樣本；1 欄：每筆一個 Feature。
    # X 是二維陣列，形狀為 (200, 1)。
    X = rng.uniform(-2, 2, size=(CONFIG["n_samples"], 1))

    # 取所有列的第 0 欄，變成一維陣列，形狀為 (200,)。
    x = X[:, 0]

    # 生成正確答案：x>0 為類別 1，其他為類別 0。
    y = (x > 0).astype(np.int64)

    print("\n資料形狀：", X.shape, y.shape)
    print("各類別數量：", np.bincount(y, minlength=2))


    # ==================== 5. 主線：固定資料切分 ====================

    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(len(X))

    n_train = int(len(X) * CONFIG["train_ratio"])
    n_validation = int(len(X) * CONFIG["validation_ratio"])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train: n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    x_train, y_train = x[train_indices], y[train_indices]
    x_validation, y_validation = x[validation_indices], y[validation_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    print("Train：", x_train.shape, y_train.shape)
    print("Validation：", x_validation.shape, y_validation.shape)
    print("Test：", x_test.shape, y_test.shape)


    # ==================== 6. 主線：觀察初始模型 ====================

    # 尚未訓練，只觀察初始參數在 Train 上的表現。
    weight = CONFIG["initial_weight"]
    bias = CONFIG["initial_bias"]

    train_p = predict_proba(x_train, weight, bias)
    train_predictions = predict_labels(train_p, threshold=CONFIG["threshold"])

    train_bce = compute_bce(y_train, train_p)
    train_acc = np.mean(train_predictions == y_train)

    print("\n初始模型 Training BCE：", train_bce)
    print("初始模型 Training Accuracy：", train_acc)



    # ==================== 7. 比較學習率 ====================

    results = {}

    for learning_rate in CONFIG["learning_rates"]:
        weight, bias, loss_history = fit_logistic_regression(
            x_train,
            y_train,
            initial_weight=CONFIG["initial_weight"],
            initial_bias=CONFIG["initial_bias"],
            learning_rate=learning_rate,
            iterations=CONFIG["iterations"],
        )

        train_p = predict_proba(x_train, weight, bias)
        validation_p = predict_proba(x_validation, weight, bias)

        train_predictions = predict_labels(
            train_p, threshold=CONFIG["threshold"]
        )
        validation_predictions = predict_labels(
            validation_p, threshold=CONFIG["threshold"]
        )

        train_bce = compute_bce(y_train, train_p)
        validation_bce = compute_bce(y_validation, validation_p)

        train_accuracy = np.mean(train_predictions == y_train)
        validation_accuracy = np.mean(
            validation_predictions == y_validation
        )

        results[learning_rate] = {
            "weight": weight,
            "bias": bias,
            "loss_history": loss_history,
            "train_bce": train_bce,
            "validation_bce": validation_bce,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        }

        print(
            f"LR={learning_rate}："
            f"Train BCE={train_bce:.6f}，"
            f"Validation BCE={validation_bce:.6f}，"
            f"Train ACC={train_accuracy:.4f}，"
            f"Validation ACC={validation_accuracy:.4f}"
        )


    # ==================== 8. 選擇模型，再做最後 Test 評估 ====================
    # rate 是函式參數；min 會逐一把 results 的學習率交進來取得比較分數。
    def get_validation_bce(rate):
        return results[rate]["validation_bce"]

    # 比較結束後才選擇；不是在每輪訓練後使用 Test。
    selected_learning_rate = min(results, key=get_validation_bce)
    selected_result = results[selected_learning_rate]
    selected_weight = selected_result["weight"]
    selected_bias = selected_result["bias"]

    test_p = predict_proba(x_test, selected_weight, selected_bias)
    test_predictions = predict_labels(test_p, threshold=CONFIG["threshold"])
    test_bce = compute_bce(y_test, test_p)
    test_accuracy = np.mean(test_predictions == y_test)

    print("\n選定 Learning Rate：", selected_learning_rate)
    print("選定 Weight：", selected_weight)
    print("選定 Bias：", selected_bias)
    print("Test BCE：", test_bce)
    print("Test Accuracy：", test_accuracy)
    # 本資料完全可分，固定 200 次更新不等於參數已收斂。



    # ==================== 選定模型的機率曲線 ====================

    # 規則排列的畫圖座標，不參與訓練。
    line_x = np.linspace(-2, 2, 200)
    line_p = predict_proba(
        line_x,
        selected_weight,
        selected_bias,
    )

    fig_probability, ax = plt.subplots()

    ax.scatter(
        x_train,
        y_train,
        alpha=0.4,
        label="Train labels",
    )

    ax.plot(
        line_x,
        line_p,
        color="tab:red",
        label="Predicted P(y=1)",
    )

    ax.axhline(
        CONFIG["threshold"],
        color="gray",
        linestyle="--",
        label="Probability threshold",
    )

    ax.axvline(
        0,
        color="black",
        linestyle=":",
        label="True boundary x=0",
    )

    ax.set_xlabel("Feature x")
    ax.set_ylabel("Label / Probability")
    ax.set_title(
        f"Logistic Regression (LR={selected_learning_rate})"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    fig_probability.tight_layout()

    figure_path = (
        Path(__file__).resolve().parent
        / "logistic_regression_probability.png"
    )
    fig_probability.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight",
    )

    print("機率曲線已儲存：", figure_path)
    plt.close(fig_probability)


    # ==================== 學習率與 Training BCE 比較 ====================

    fig_loss, ax = plt.subplots()

    for learning_rate in CONFIG["learning_rates"]:
        history = results[learning_rate]["loss_history"]
        rounds = np.arange(len(history))

        ax.plot(
            rounds,
            history,
            label=f"Learning rate={learning_rate}",
        )

    ax.set_xlabel("Updates")
    ax.set_ylabel("Training BCE")
    ax.set_title("Learning Rate Comparison")
    ax.legend()

    fig_loss.tight_layout()

    loss_path = (
        Path(__file__).resolve().parent
        / "logistic_regression_learning_rates.png"
    )

    fig_loss.savefig(loss_path, dpi=150, bbox_inches="tight")
    print("Loss 比較圖已儲存：", loss_path)

    plt.close(fig_loss)


if __name__ == "__main__":
    main()
