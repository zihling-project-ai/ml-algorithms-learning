"""輔助學習：Sigmoid、BCE 展開與有限差分，不使用正式 Test。"""
import numpy as np
from logistic_regression import sigmoid, predict_proba, compute_bce, compute_gradients

def estimate_gradients(x, y_true, weight, bias, step):
    # ① 原本的機率與 BCE。
    original_p = predict_proba(x, weight, bias)
    original_loss = compute_bce(y_true, original_p)

    # ② 只將 Weight 增加一點，看看 BCE 改變多少。
    weight_probe_p = predict_proba(x, weight + step, bias)
    weight_probe_loss = compute_bce(y_true, weight_probe_p)

    weight_loss_change = weight_probe_loss - original_loss
    grad_weight = weight_loss_change / step

    # ③ 回到原本參數，只將 Bias 增加一點。
    bias_probe_p = predict_proba(x, weight, bias + step)
    bias_probe_loss = compute_bce(y_true, bias_probe_p)

    bias_loss_change = bias_probe_loss - original_loss
    grad_bias = bias_loss_change / step

    return grad_weight, grad_bias


def main():
    # 1. Sigmoid 是固定轉換，不依本批最大最小值重新縮放。
    z_demo = np.array([-2.0, 0.0, 2.0])
    print("Sigmoid：", sigmoid(z_demo))

    # 2. p 是類別 1 機率；類別 0 的機率是 1-p。
    y_demo = np.array([1, 0])
    p_demo = np.array([0.9, 0.1])
    safe_p = np.clip(p_demo, 1e-12, 1 - 1e-12)
    term_1 = y_demo * np.log(safe_p)
    term_0 = (1 - y_demo) * np.log(1 - safe_p)
    losses = -(term_1 + term_0)
    print("類別 1 項：", term_1)
    print("類別 0 項：", term_0)
    print("各筆 Loss：", losses)
    print("平均 BCE：", np.mean(losses))
    print("較差預測 BCE：", compute_bce(y_demo, np.array([0.1, 0.9])))

    # 3. zip 把同位置的答案與機率配成一對。
    for y_true, p in zip(y_demo, p_demo):
        if y_true == 1:
            loss = -np.log(p)
        else:
            loss = -np.log(1 - p)
        print("逐筆：", y_true, p, "Loss：", loss)

    # 4. 同一筆資料對照有限差分與解析梯度。
    x_one = np.array([1.0])
    y_one = np.array([1])
    weight, bias = 0.0, 0.0
    p = predict_proba(x_one, weight, bias)
    approx = estimate_gradients(x_one, y_one, weight, bias, step=0.001)
    exact = compute_gradients(x_one, y_one, p)
    print("有限差分梯度：", approx)
    print("解析梯度：", exact)


if __name__ == "__main__":
    main()
