"""L07 輔助學習：原始公式展開、單溫度與多特徵逐步觀察。
保留整理前的教學內容；正式實驗請執行 gaussian_naive_bayes.py。
直接密度示例僅適用本例的小數值，Log 穩定版另有完整對照。
"""
import numpy as np


def run_learning_examples():
    # ==================== 正式實驗設定 ====================

    CONFIG = {
        "data_seed": 42,
        "split_seed": 42,
        "n_normal": 120,
        "n_abnormal": 80,
        "train_ratio": 0.6,
        "validation_ratio": 0.2,
        "variance_floor": 1e-6,
    }



    # ==================== A. 模型核心：展開公式示例 ====================

    # ==================== 1. 訓練資料：溫度與真實類別 ====================

    # 每個位置是一筆設備資料；同位置的溫度與標籤配對。
    # 本輪六筆都是教學用訓練資料，尚未進行 Train／Validation／Test 實驗。
    temperatures = np.array([30., 32., 34., 36., 40., 44.])
    y = np.array([0, 0, 0, 0, 1, 1])

    # mask：用標籤選出各類對應的溫度，不是把溫度拿來當類別編號。
    normal_temperatures = temperatures[y == 0]      # [30, 32, 34, 36]
    abnormal_temperatures = temperatures[y == 1]    # [40, 44]


    # ==================== 多特徵資料：溫度、振動 ====================

    X_demo = np.array([
        [30.0, 0.02],
        [32.0, 0.03],
        [34.0, 0.02],
        [36.0, 0.03],
        [40.0, 0.08],
        [44.0, 0.10],
    ])

    query_point = np.array([40.0, 0.09])

    print("多特徵資料形狀：", X_demo.shape)
    print("新設備形狀：", query_point.shape)

    # ==================== 2. 訓練統計量：Prior、平均值、變異數 ====================

    # 公式：Prior_k = P(y=k) = 第 k 類筆數 n_k / 全部筆數 n
    # P：機率；k：類別編號。此時尚未使用新設備溫度。
    normal_prior = len(normal_temperatures) / len(y)       # 4/6
    abnormal_prior = len(abnormal_temperatures) / len(y)   # 2/6

    # 公式：μ_k = (1/n_k) × Σ x_i，Σ 表示將該類的數值加總。
    # μ（mu）：該類分布的中心；本例單位為 °C。
    normal_mean = np.mean(normal_temperatures)       # 33
    abnormal_mean = np.mean(abnormal_temperatures)   # 42

    # 公式：v_k = σ_k² = (1/n_k) × Σ (x_i - μ_k)²
    # v：變異數，描述分散程度；單位為 °C²。
    # np.var 預設 ddof=0，除以該類筆數 n_k。
    # Normal：差距 [-3,-1,1,3] → 平方 [9,1,1,9] → 平均 5。
    # Abnormal：差距 [-2,2] → 平方 [4,4] → 平均 4。
    normal_variance = np.var(normal_temperatures)
    abnormal_variance = np.var(abnormal_temperatures)

    print("Normal Prior／平均／變異數：",
          normal_prior, normal_mean, normal_variance)
    print("Abnormal Prior／平均／變異數：",
          abnormal_prior, abnormal_mean, abnormal_variance)


    # ==================== 3. 新資料：相對各類中心的偏離程度 ====================

    # 新設備的待判斷溫度；這是單筆推論示例，不是獨立 Test 評估。
    query_t = 40.0

    # 公式：scaled_distance_k = (x - μ_k)² / v_k
    # x 為 query_t。這還不是機率，只是相對分散程度的平方距離。
    normal_scaled_distance = (query_t - normal_mean) ** 2 / normal_variance
    abnormal_scaled_distance = (query_t - abnormal_mean) ** 2 / abnormal_variance

    print("相對 Normal 的平方距離：", normal_scaled_distance)       # 49/5 = 9.8
    print("相對 Abnormal 的平方距離：", abnormal_scaled_distance)   # 4/4 = 1.0


    # ==================== 4. Likelihood：代入各類的常態分布密度 ====================

    # 模型假設：各類溫度可以用鐘形的常態分布近似。
    # 不是把原始資料強制變成常態，也不是由少量樣本證明分布必然為鐘形。
    #
    # 完整公式：
    # f(x | y=k) = 1/√(2πv_k) × exp(-(x - μ_k)² / (2v_k))
    #            = height_k × decay_k
    #
    # f(x | y=k)：假設類別為 k，x 位置的機率密度。
    # 「|」讀作「在……條件下」。π ≈ 3.14159；exp(a) = e^a。
    # 這是密度高度，不是 P(y=k | x)，也不是連續溫度恰等於 x 的機率。

    # 衰減部分：exp(-0.5 × scaled_distance)，越偏離中心越小。
    # -0.5 是常態分布公式的一部分，使 v_k 對應指定變異數，不是超參數。
    normal_decay = np.exp(-0.5 * normal_scaled_distance)
    abnormal_decay = np.exp(-0.5 * abnormal_scaled_distance)

    # 高度部分：1/√(2πv_k)，使完整曲線下面積為 1。
    normal_height = 1 / np.sqrt(2 * np.pi * normal_variance)
    abnormal_height = 1 / np.sqrt(2 * np.pi * abnormal_variance)

    # 在本次固定 x、比較類別的用途下，將這些類別條件密度稱為 Likelihood。
    normal_likelihood = normal_height * normal_decay
    abnormal_likelihood = abnormal_height * abnormal_decay

    print("Normal Likelihood：", normal_likelihood)       # 約 0.001329
    print("Abnormal Likelihood：", abnormal_likelihood)   # 約 0.120985


    # ==================== 5. 結合 Prior 與 Likelihood：得到類別分數 ====================

    # 公式：score_k = P(y=k) × f(x | y=k)
    # 同時考慮類別原本有多常見，以及新溫度在該類中有多常見。
    # score 還不是加總為 1 的類別機率。
    normal_score = normal_prior * normal_likelihood
    abnormal_score = abnormal_prior * abnormal_likelihood

    print("Normal 分數：", normal_score)
    print("Abnormal 分數：", abnormal_score)


    # ==================== 6. Posterior：分數正規化為後驗機率 ====================

    # Bayes 公式（本例兩類）：
    # P(y=k | x) = P(y=k) × f(x | y=k) / f(x)
    # f(x) = score_0 + score_1
    #
    # P(y=k | x)：看到溫度 x 後，在模型假設下屬於第 k 類的機率。
    # f(x)：新溫度在全部類別混合分布中的密度；兩類共用相同分母。
    # 本例分數皆非零，因此可直接相除。
    total_score = normal_score + abnormal_score
    normal_posterior = normal_score / total_score
    abnormal_posterior = abnormal_score / total_score

    print("Normal 後驗機率：", normal_posterior)       # 約 2.15%
    print("Abnormal 後驗機率：", abnormal_posterior)   # 約 97.85%


    # ==================== 7. 分類：選分數最大的類別 ====================

    # 公式：ŷ = argmax_k score_k，也等於 argmax_k P(y=k | x)。
    # ŷ：預測類別；argmax：回傳最大值的位置／對應類別。
    # 這裡明確按照 [Normal=0, Abnormal=1] 排列，同票時選 0。
    # 除以共用總分不會改變排序，只做分類時不必先正規化。
    prediction = np.argmax([normal_score, abnormal_score])
    print("預測類別：", prediction)   # 1，Abnormal

    # 後驗機率依賴小樣本統計與常態假設，不代表真實世界一定如此。


    # ==================== B. 函式版：相同模型流程 ====================

    # -------------------- 數值穩定性備註：與模型定義區分 --------------------
    # 1. 變異數下限：v_safe = max(v_raw, variance_floor)。
    #    這是額外保護，避免 v=0 時除以零；會改變過小的變異數及相應預測。
    #    原始 Gaussian 公式要求 v>0，並未規定一律使用 1e-6。
    #    下限與 Feature 尺度有關，本例單位為 °C²；它不是通用最佳值。
    #
    # 2. Log 計算：對原公式作等價改寫，理論上不改變類別分數排序。
    #    log_score_k = log(Prior_k) + log f(x | y=k)
    #    log f(x | y=k) = -0.5 × [log(2πv_k) + (x-μ_k)²/v_k]
    #    必須直接計算這個 Log 密度，不能等密度已下溢成 0 才 np.log。
    #    Log 分數不是後驗機率；本版使用以下穩定正規化：
    #    shifted_k = log_score_k - max(log_scores)
    #    posterior_k = exp(shifted_k) / Σ exp(shifted_j)
    #    【已實作】predict_one 直接計算 Log 密度，再做穩定正規化。
    #
    # 區別：變異數下限調整模型統計量；Log 域計算改變運算表示方式。

    # ==================== 訓練：估計各類分布 ====================

    def fit_gaussian_naive_bayes(temperatures, y, variance_floor=1e-6):
        classes = np.unique(y)

        priors = []
        means = []
        variances = []

        for label in classes:
            # 只取出目前這個類別的溫度。
            members = temperatures[y == label]

            # 將先前分開寫的計算，改成每個類別各做一次。
            priors.append(len(members) / len(y))
            means.append(np.mean(members))
            # 【模型核心】原始類別變異數：平均平方偏差。
            raw_variance = np.var(members)

            # 【數值保護】套用下限；本例 5、4 不受影響。
            safe_variance = np.maximum(raw_variance, variance_floor)

            variances.append(safe_variance)

        return {
            "classes": classes,
            "priors": np.array(priors),
            "means": np.array(means),
            "variances": np.array(variances),
        }


    # ==================== 預測：單筆溫度（Log 穩定版） ====================

    def predict_one_gaussian_naive_bayes(query_t, model):
        priors = model["priors"]
        means = model["means"]
        variances = model["variances"]

        # 【模型公式的等價改寫】直接計算 Gaussian Log 密度。
        # log f(x|k) = -0.5 × [log(2πv_k) + (x-μ_k)²/v_k]
        # 不先算原密度，避免密度已下溢成 0 才取 Log。
        log_likelihoods = -0.5 * (
            np.log(2 * np.pi * variances)
            + (query_t - means) ** 2 / variances
        )
        log_scores = np.log(priors) + log_likelihoods

        # 【數值穩定性】所有類別共減最大值，保持比例與排序。
        # 最大 shifted_score 為 0，所以至少有一個 relative_score 為 1。
        shifted_scores = log_scores - np.max(log_scores)
        relative_scores = np.exp(shifted_scores)
        posteriors = relative_scores / np.sum(relative_scores)

        # Log 保留大小順序；先找位置，再取對應的真實類別。
        best_index = np.argmax(log_scores)
        prediction = model["classes"][best_index]

        return prediction, posteriors


    # ==================== C. 主線呼叫：訓練後預測 ====================

    # 預設 variance_floor=1e-6 是數值保護設定，不是 Gaussian 密度公式的常數。
    model = fit_gaussian_naive_bayes(temperatures, y)

    prediction, posteriors = predict_one_gaussian_naive_bayes(
        40.0,
        model,
    )

    print("類別順序：", model["classes"])
    print("後驗機率：", posteriors)
    print("預測類別：", prediction)


    # ==================== 預測：多筆溫度 ====================

    def predict_gaussian_naive_bayes(query_temperatures, model):
        predictions = []
        posterior_rows = []

        for query_t in query_temperatures:
            prediction, posteriors = predict_one_gaussian_naive_bayes(
                query_t,
                model,
            )

            predictions.append(prediction)
            posterior_rows.append(posteriors)

        return np.array(predictions), np.array(posterior_rows)


    query_temperatures = np.array([32.0, 40.0, 44.0])

    predictions, posteriors = predict_gaussian_naive_bayes(
        query_temperatures,
        model,
    )

    print("查詢溫度：", query_temperatures)
    print("預測類別：", predictions)
    print("各筆後驗機率：")
    print(posteriors)

    print("預測形狀：", predictions.shape)
    print("後驗機率形狀：", posteriors.shape)

    assert predictions.shape == query_temperatures.shape


    # ==================== D. 多特徵訓練觀察：保留上方單溫度版本 ====================
    # 本區觀察逐欄統計量；E 區塊再結合各 Feature 進行單筆預測。
    # 新增獨立函式名稱，不重新定義上方 fit_gaussian_naive_bayes。

    def fit_gaussian_naive_bayes_multifeature(X, y, variance_floor=1e-6):
        classes = np.unique(y)
        priors = []
        means = []
        variances = []

        print("\n【多特徵訓練觀察】類別順序：", classes)

        for label in classes:
            mask = y == label
            members = X[mask]

            # Prior_k = n_k / n：每類一個比例。
            prior = len(members) / len(y)

            # μ_kj = (1/n_k) × Σ_i X_ij
            # axis=0：跨樣本逐欄計算，不把溫度與振動混在一起平均。
            feature_means = np.mean(members, axis=0)

            # v_kj = (1/n_k) × Σ_i (X_ij - μ_kj)²
            raw_variances = np.var(members, axis=0)
            # 【數值保護】每一欄各自套用變異數下限。
            safe_variances = np.maximum(raw_variances, variance_floor)

            priors.append(prior)
            means.append(feature_means)
            variances.append(safe_variances)

            print("\n目前類別：", label)
            print("選取 mask：", mask)
            print("該類資料（欄：溫度、振動）：\n", members)
            print("該類筆數／全部筆數：", len(members), "/", len(y))
            print("Prior：", prior)
            print("axis=0 逐欄平均：", feature_means)
            print("axis=0 逐欄原始變異數：", raw_variances)
            print("套用下限後變異數：", safe_variances)
            print("目前 priors 清單：", priors)
            print("目前 means 清單：", means)
            print("目前 variances 清單：", variances)

        return {
            "classes": classes,
            "priors": np.array(priors),
            "means": np.array(means),
            "variances": np.array(variances),
        }


    # 每列一筆設備，欄位固定為 [溫度, 振動]。
    X_demo = np.array([
        [30.0, 0.02],
        [32.0, 0.03],
        [34.0, 0.02],
        [36.0, 0.03],
        [40.0, 0.08],
        [44.0, 0.10],
    ])
    query_point = np.array([40.0, 0.09])

    # 不覆寫上方單特徵 model，另存多特徵統計量。
    multi_model = fit_gaussian_naive_bayes_multifeature(X_demo, y)

    print("\n回傳 model 的類別：", multi_model["classes"])
    print("回傳 model 的 Prior：", multi_model["priors"])
    print("回傳 model 的平均值（列：類別；欄：Feature）：\n", multi_model["means"])
    print("回傳 model 的變異數（列：類別；欄：Feature）：\n", multi_model["variances"])
    print("平均值 shape：", multi_model["means"].shape)
    print("變異數 shape：", multi_model["variances"].shape)

    assert multi_model["means"].shape == (2, 2)
    assert multi_model["variances"].shape == (2, 2)



    # ==================== E. 多特徵單筆預測：結合溫度與振動 ====================

    def predict_one_gaussian_naive_bayes_multifeature(point, model):
        means = model["means"]
        variances = model["variances"]

        # 輸入 point：(Features,)；means、variances：(類別數, Features)。
        # Broadcasting：同一筆 point 分別與每一類的統計量計算。
        # 每格的公式：log f(x_j|y=k)
        # = -0.5 × [log(2πv_kj) + (x_j-μ_kj)²/v_kj]
        log_likelihoods = -0.5 * (
            np.log(2 * np.pi * variances)
            + (point - means) ** 2 / variances
        )

        # 【模型核心】條件獨立近似：score_k = Prior_k × Π_j f(x_j|y=k)
        # Π 表示相乘；取 Log 後：log_score_k = log Prior_k + Σ_j log f(x_j|y=k)。
        # log_likelihoods 每列為一類、每欄為一個 Feature。
        # axis=1：橫向加總同類的溫度、振動證據，得到每類一個分數。
        log_scores = (
            np.log(model["priors"])
            + np.sum(log_likelihoods, axis=1)
        )

        # 【數值穩定性】平移後正規化；模型規則不變。
        shifted_scores = log_scores - np.max(log_scores)
        relative_scores = np.exp(shifted_scores)
        posteriors = relative_scores / np.sum(relative_scores)

        best_index = np.argmax(log_scores)
        prediction = model["classes"][best_index]
        return prediction, posteriors


    multi_prediction, multi_posteriors = (
        predict_one_gaussian_naive_bayes_multifeature(query_point, multi_model)
    )

    print("\n新設備 [溫度, 振動]：", query_point)
    print("多特徵類別順序：", multi_model["classes"])
    print("多特徵後驗機率：", multi_posteriors)
    print("多特徵預測類別：", multi_prediction)

    assert multi_posteriors.shape == multi_model["classes"].shape
    assert np.isfinite(multi_posteriors).all()
    assert np.isclose(np.sum(multi_posteriors), 1.0)
    # 極高後驗機率依賴小樣本與分布假設，並不代表現實中絕對確定。


    # ==================== F. 多特徵批次預測 ====================

    def predict_gaussian_naive_bayes_multifeature(X_new, model):
        predictions = []
        posterior_rows = []

        for point in X_new:
            prediction, posteriors = (
                predict_one_gaussian_naive_bayes_multifeature(
                    point,
                    model,
                )
            )

            predictions.append(prediction)
            posterior_rows.append(posteriors)

        return np.array(predictions), np.array(posterior_rows)


    query_points = np.array([
        [32.0, 0.025],
        [40.0, 0.09],
        [44.0, 0.10],
    ])

    multi_predictions, multi_posteriors = (
        predict_gaussian_naive_bayes_multifeature(
            query_points,
            multi_model,
        )
    )

    print("多筆設備：")
    print(query_points)

    print("預測類別：", multi_predictions)
    print("後驗機率：")
    print(multi_posteriors)

    print("預測 shape：", multi_predictions.shape)
    print("後驗機率 shape：", multi_posteriors.shape)

    assert multi_predictions.shape == (len(query_points),)
    assert multi_posteriors.shape == (
        len(query_points),
        len(multi_model["classes"]),
    )


if __name__ == "__main__":
    run_learning_examples()
