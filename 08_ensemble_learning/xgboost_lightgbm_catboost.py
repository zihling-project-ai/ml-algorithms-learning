"""L06｜Boosting 工具的廣告案例。
實際執行範圍僅 XGBoost：資料 → 固定切分 → 品牌編碼 → 訓練 → Validation → 新活動逐輪預測。
LightGBM、CatBoost 僅做概念比較，另見 BOOSTING_TOOLS_COMPARISON.md。
合成資料僅用於學習，不是廣告成效的因果模型；Test 保留未評估。
"""
import numpy as np
import xgboost as xgb
import json

# ==================== 1. 設定 ====================

CONFIG = {
    "data_seed": 42,
    "split_seed": 42,
    "n_samples": 300,
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "xgb_rounds": 100,
    "xgb_learning_rate": 0.1,
    "xgb_max_depth": 3,
    "xgb_reg_lambda": 1.0,
}

def add_brand_column(numeric, brands):
    """本案例只有紅／藍兩種品牌，新增是否紅牌的 0／1 欄位。"""
    is_red = (brands == "紅牌").astype(float)
    return np.column_stack((numeric, is_red))


def main():
    # ==================== 2. 建立虛構廣告資料 ====================

    rng = np.random.default_rng(CONFIG["data_seed"])
    n = CONFIG["n_samples"]

    budget = rng.uniform(1, 15, size=n)       # 單位：萬元
    discount = rng.choice([0.0, 0.1, 0.2], size=n)
    brands = rng.choice(["紅牌", "藍牌"], size=n)
    noise = rng.normal(0, 5, size=n)

    # 人為設定的品牌差異，只用來建立學習案例。
    brand_effect = np.where(brands == "紅牌", 8.0, 0.0)

    # 合成銷售規則；模型不會直接取得這條公式。
    y = (
        20
        + 7 * budget
        + 100 * discount
        + brand_effect
        + noise
    )

    X_numeric = np.column_stack((budget, discount))

    # ==================== 3. 固定切分編號 ====================

    rng_split = np.random.default_rng(CONFIG["split_seed"])
    indices = rng_split.permutation(n)

    n_train = int(n * CONFIG["train_ratio"])
    n_validation = int(n * CONFIG["validation_ratio"])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train:n_train + n_validation]
    test_indices = indices[n_train + n_validation:]

    # 同一組編號，同時取出數值、品牌與答案。
    X_train = X_numeric[train_indices]
    brands_train = brands[train_indices]
    y_train = y[train_indices]

    X_validation = X_numeric[validation_indices]
    brands_validation = brands[validation_indices]
    y_validation = y[validation_indices]

    X_test = X_numeric[test_indices]
    brands_test = brands[test_indices]
    y_test = y[test_indices]

    print("Train：", X_train.shape, brands_train.shape, y_train.shape)
    print(
        "Validation：",
        X_validation.shape,
        brands_validation.shape,
        y_validation.shape,
    )
    print("Test：", X_test.shape, brands_test.shape, y_test.shape)

    print("第一筆 Train：")
    print("預算、折扣：", X_train[0])
    print("品牌：", brands_train[0])
    print("銷售量：", y_train[0])

    assert len(X_train) == len(brands_train) == len(y_train)

    # ==================== 4. 待預測的新活動 ====================

    new_numeric = np.array([[7.0, 0.1]])
    new_brands = np.array(["紅牌"])

    # ==================== 5. XGBoost：準備輸入 ====================


    X_train_xgb = add_brand_column(X_train, brands_train)
    X_validation_xgb = add_brand_column(
        X_validation, brands_validation,
    )
    X_new_xgb = add_brand_column(new_numeric, new_brands)

    print("XGBoost 第一筆 Train：", X_train_xgb[0])

    # DMatrix 是 XGBoost 接收資料的容器，不是訓練步驟。
    dtrain = xgb.DMatrix(X_train_xgb, label=y_train)
    dvalidation = xgb.DMatrix(X_validation_xgb)
    dnew = xgb.DMatrix(X_new_xgb)

    params = {
        "objective": "reg:squarederror",
        "eta": CONFIG["xgb_learning_rate"],
        "max_depth": CONFIG["xgb_max_depth"],
        "reg_lambda": CONFIG["xgb_reg_lambda"],
        "tree_method": "hist",
        "seed": CONFIG["data_seed"],
        "nthread": 1,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=CONFIG["xgb_rounds"],
    )

    validation_predictions = model.predict(dvalidation)
    new_predictions = model.predict(dnew)

    validation_mse = np.mean(
        (y_validation - validation_predictions) ** 2
    )

    print("XGBoost Validation MSE：", validation_mse)
    print("XGBoost Validation RMSE（件）：", np.sqrt(validation_mse))
    print("新活動預測銷售量：", new_predictions[0])

    assert validation_predictions.shape == y_validation.shape

    # ==================== 觀察新活動的逐輪預測 ====================

    # 只讀取已訓練模型，不重新訓練。
    # iteration_range=(0, 0) 代表完整模型，不能拿來當零輪。
    # 此處是單一輸出平方誤差回歸，且未另外指定 base_margin。
    # 從已訓練模型讀取起始固定預測。
    model_config = json.loads(model.save_config())

    base_score_text = model_config[
        "learner"
    ]["learner_model_param"]["base_score"]

    # 相容起始值以單一數字或陣列儲存的格式。
    base_score = float(
        np.asarray(json.loads(base_score_text)).reshape(-1)[0]
    )

    print(f"第 0 輪：預測 {base_score:.4f} 件")

    previous_prediction = base_score
    previous_round = 0

    # 不超過已訓練輪數；確保 Config 改動後仍可觀察完整模型。
    checkpoint_rounds = sorted({
        r for r in (1, 10, CONFIG["xgb_rounds"])
        if 0 < r <= CONFIG["xgb_rounds"]
    })
    for rounds in checkpoint_rounds:
        prediction = float(
            model.predict(
                dnew,
                iteration_range=(0, rounds),
            )[0]
        )

        change = prediction - previous_prediction

        print(
            f"第 {rounds} 輪：預測 {prediction:.4f} 件；"
            f"第 {previous_round + 1}～{rounds} 輪"
            f"合計修正 {change:+.4f} 件"
        )

        previous_prediction = prediction
        previous_round = rounds


if __name__ == "__main__":
    main()
