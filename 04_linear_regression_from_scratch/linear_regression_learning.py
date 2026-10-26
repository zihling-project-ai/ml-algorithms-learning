"""輔助學習：兩筆可手算資料、有限差分、解析梯度與訓練前後比較。

獨立執行本檔不會啟動主線的 200 筆資料實驗。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from linear_regression import predict, compute_mse, compute_gradients, fit_linear_regression

LEARNING_CONFIG = {
    "initial_weight": 1.0,
    "initial_bias": 2.5,
    "difference_step": 0.0001,  # 試探距離，不是正式更新參數的學習率
    "learning_rate": 0.1,
    "iterations": 200,
}

def estimate_gradients(x, y_true, weight, bias, step):
    """有限差分：試調參數，用前後 MSE 估計變化率。"""
    loss_before = compute_mse(
        y_true, predict(x, weight, bias)
    )

    loss_probe_weight = compute_mse(
        y_true, predict(x, weight + step, bias)
    )
    loss_probe_bias = compute_mse(
        y_true, predict(x, weight, bias + step)
    )

    grad_weight = (loss_probe_weight - loss_before) / step
    grad_bias = (loss_probe_bias - loss_before) / step

    return grad_weight, grad_bias



def main():



    x_demo = np.array([1.0, 2.0])
    y_demo = np.array([3.0, 5.0])

    # 參考答案為 Weight＝2、Bias＝1，不直接交給訓練函式。
    initial_weight = LEARNING_CONFIG["initial_weight"]
    initial_bias = LEARNING_CONFIG["initial_bias"]

    approx_weight, approx_bias = estimate_gradients(
        x_demo,
        y_demo,
        initial_weight,
        initial_bias,
        LEARNING_CONFIG["difference_step"],
    )

    initial_predictions = predict(
        x_demo, initial_weight, initial_bias
    )

    exact_weight, exact_bias = compute_gradients(
        x_demo, y_demo, initial_predictions
    )

    print("有限差分梯度：", approx_weight, approx_bias)
    print("解析梯度：", exact_weight, exact_bias)


    # ==================== 6. 呼叫訓練函式 ====================

    weight, bias, loss_history = fit_linear_regression(
        x_demo,
        y_demo,
        initial_weight=initial_weight,
        initial_bias=initial_bias,
        learning_rate=LEARNING_CONFIG["learning_rate"],
        iterations=LEARNING_CONFIG["iterations"],
    )

    predictions = predict(x_demo, weight, bias)

    print("\n最終 Weight：", weight)
    print("最終 Bias：", bias)
    print("起始 MSE：", loss_history[0])
    print("最終 MSE：", loss_history[-1])
    print("最終預測：", predictions)
    print("真實答案：", y_demo)


    # ==================== 7. 查看直線與 Loss 曲線 ====================

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 左圖：資料點、初始直線、學到的直線。
    line_x = np.linspace(0.5, 2.5, 100)

    axes[0].scatter(
        x_demo, y_demo, color="black", label="True data", zorder=3
    )
    axes[0].plot(
        line_x,
        predict(line_x, initial_weight, initial_bias),
        linestyle="--",
        label="Initial line",
    )
    axes[0].plot(
        line_x,
        predict(line_x, weight, bias),
        label="Learned line",
    )
    axes[0].set_xlabel("Feature x")
    axes[0].set_ylabel("Target y")
    axes[0].set_title("Before and After Training")
    axes[0].legend()

    # 右圖：第 0 輪到第 200 輪的 MSE。
    rounds = np.arange(len(loss_history))

    axes[1].plot(rounds, loss_history)
    axes[1].set_xlabel("Updates")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Training Loss")

    fig.tight_layout()
    figure_path = Path(__file__).resolve().parent / "linear_regression_learning_demo.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("輔助練習圖：", figure_path)




if __name__ == "__main__":
    main()
