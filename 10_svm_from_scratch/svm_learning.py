"""Linear SVM 輔助學習：手算候選對照與六筆訓練示範。
獨立執行；不會在正式主線訓練時執行。
"""
import numpy as np

from svm import decision_function, compute_svm_loss, fit_linear_svm

CONFIG = {
    "learning_rate": 0.05,
    "iterations": 500,
    "demo_C": 1.0,
    "candidate_settings": (
        {"weight": 0.5, "bias": -18.0, "C": 1.0},
        {"weight": 0.25, "bias": -9.0, "C": 1.0},
        {"weight": 0.25, "bias": -9.0, "C": 0.1},
    ),
}


def run_learning_examples():
    # 保留原本二維分類分數例子；此處為標準化座標。
    X_coordinate_demo = np.array([[-0.6, -0.2], [0.6, 0.3]])
    demo_scores = decision_function(
        X_coordinate_demo, np.array([1.0, 0.0]), bias=0.0,
    )
    print("二維座標示例分類分數：", demo_scores)

    # 重新教學使用的一維溫度例子，每列一筆，單位為 °C。
    X_demo = np.array([[34.0], [38.0]])
    y_signed = np.array([-1, 1])

    for setting in CONFIG["candidate_settings"]:
        weights = np.array([setting["weight"]])
        bias = setting["bias"]
        C = setting["C"]

        scores = decision_function(X_demo, weights, bias)
        predictions = np.where(scores >= 0, 1, -1)
        total, penalty, hinge = compute_svm_loss(
            X_demo, y_signed, weights, bias, C,
        )

        print(f"\nw={weights[0]}，b={bias}，C={C}")
        print("分類分數：", scores)
        print("預測類別（-1=Normal、+1=Abnormal）：", predictions)
        print("標籤 × 分數：", y_signed * scores)
        print("權重代價：", penalty)
        print("平均 Hinge Loss：", hinge)
        print("總目標：", total)

    # 這是公式與程式的一致性檢查，不是模型的泛化評估。
    assert np.allclose(demo_scores, [-0.6, 0.6])
    total, penalty, hinge = compute_svm_loss(
        X_demo, y_signed, np.array([0.5]), -18.0, C=1.0,
    )
    assert np.allclose([total, penalty, hinge], [0.125, 0.125, 0.0])
    total, penalty, hinge = compute_svm_loss(
        X_demo, y_signed, np.array([0.25]), -9.0, C=0.1,
    )
    assert np.allclose([total, penalty, hinge], [0.08125, 0.03125, 0.5])
    print("\n手算與程式結果一致。")

    # 六筆資料只驗證訓練接線，尚非正式 Train／Validation／Test 實驗。
    X_devices = np.array([
        [30.0, 0.02], [32.0, 0.04], [34.0, 0.03],
        [38.0, 0.04], [40.0, 0.06], [42.0, 0.03],
    ])
    device_labels = np.array([-1, -1, -1, 1, 1, 1])
    feature_means = np.mean(X_devices, axis=0)
    feature_stds = np.std(X_devices, axis=0)
    # 常數欄以 1 代替零標準差，避免除以零；本例兩欄皆非零。
    safe_stds = np.where(feature_stds > 0, feature_stds, 1.0)
    X_scaled = (X_devices - feature_means) / safe_stds

    weights, bias, loss_history = fit_linear_svm(
        X_scaled, device_labels, C=CONFIG["demo_C"],
        learning_rate=CONFIG["learning_rate"], iterations=CONFIG["iterations"],
    )
    scores = decision_function(X_scaled, weights, bias)
    predictions = np.where(scores >= 0, 1, -1)
    print("\n【六筆訓練接線示範】")
    print("標準化平均：", feature_means)
    print("標準化標準差：", safe_stds)
    print("學到的 weights：", weights)
    print("學到的 bias：", bias)
    print("起始／最終總目標：", loss_history[0], loss_history[-1])
    print("預測類別：", predictions)
    print("教學資料 Training Accuracy：", np.mean(predictions == device_labels))
    assert np.isfinite(loss_history).all()
    assert len(loss_history) == CONFIG["iterations"] + 1




if __name__ == "__main__":
    run_learning_examples()
