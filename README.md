# ML Algorithms Learning

Educational Machine Learning implementations with Python, NumPy and Matplotlib. 學習主線涵蓋資料建立、演算法核心、實驗比較、未見資料評估及數學圖解。

核心演算法由 NumPy 實作，未使用 Scikit-learn 模型或切分工具。`08_ensemble_learning/xgboost_lightgbm_catboost.py` 是另列的 **XGBoost 套件示範**，不屬於 From-Scratch 實作；LightGBM、CatBoost 僅做概念比較。

本專案使用合成資料，不是生產環境的預測系統。個別 Lab 核心理解通過與程式可執行，不代表全部數學已能獨立推導，亦不代表 Phase 01 整合驗收已完成。

## Repository 結構

```text
01_numpy_fundamentals/                 合成資料與 NumPy
02_knn_from_scratch/                   KNN 分類
03_kmeans_from_scratch/                K-Means 分群
04_linear_regression_from_scratch/     線性回歸與梯度
05_logistic_regression_from_scratch/   Sigmoid、BCE 與分類
06_decision_tree_from_scratch/         Gini、切分與遞迴樹
07_random_forest_from_scratch/         Bootstrap、節點隨機欄位與投票
08_ensemble_learning/                 Bagging、AdaBoost、Gradient Boosting
09_gaussian_naive_bayes_from_scratch/  Prior、Gaussian Likelihood 與 Posterior
10_svm_from_scratch/                  Linear Soft-Margin SVM
11_pca_from_scratch/                   PCA、投影與近似還原
ML_REVIEW.md                         演算法總複習
MATH_DERIVATION_NOTES.md              共用數學筆記
```

SVM 資料夾名稱已統一為 `10_svm_from_scratch`。各章的 `*_learning.py` 保存小型推導範例，主程式聚焦實驗；THEORY_QA.md 記錄回答與修正，不把所有學習歷程當成正式驗證證據。

## 安裝與執行

已驗證環境：**CPython 3.14.7、macOS Apple Silicon、NumPy 2.5.4、Matplotlib 3.11.2**。其他 Python 版本與作業系統尚未實測，不保證相同二進位依賴與浮點結果。

在 Repository 根目錄：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 02_knn_from_scratch/knn_classification.py
python 11_pca_from_scratch/pca.py
```

選用的 XGBoost 示範：

```bash
python -m pip install -r requirements-optional.txt
python 08_ensemble_learning/xgboost_lightgbm_catboost.py
```

需要精確重建本輪完整套件版本時，可改安裝 `requirements-lock.txt`；該檔包含選用 XGBoost 與其 SciPy 依賴，是實測環境的版本快照，不是跨平台帶雜湊的通用鎖檔。

程式以 `Path(__file__)` 決定圖檔路徑，輸出至各章資料夾，重跑會更新同名圖檔。06／07／08 沿用相鄰章節的核心函式，需保留資料夾結構。所有範例均有入口保護，匯入不會啟動實驗。

在無圖形介面的環境可用：

```bash
MPLBACKEND=Agg python 10_svm_from_scratch/svm.py
```

直接執行範例會更新圖檔。公開版保留各章內建的 assert 與學習範例，未附發布前另行使用的測試／稽核工具。

## 驗證紀錄與範例執行

發布前另以專案外工具完成 25 項針對性測試與完整依賴環境的 19 支範例檢查；工具不包含在目前精簡版中。各章程式仍保留形狀、數值與學習示例的檢查。

可逐章執行下列範例；未安裝選用依賴時先略過 XGBoost：

```bash
python 01_numpy_fundamentals/synthetic_classification.py
python 02_knn_from_scratch/knn_classification.py
python 03_kmeans_from_scratch/kmeans_clustering.py
python 04_linear_regression_from_scratch/linear_regression.py
python 05_logistic_regression_from_scratch/logistic_regression.py
python 06_decision_tree_from_scratch/decision_tree.py
python 07_random_forest_from_scratch/random_forest.py
python 08_ensemble_learning/ensemble_learning.py
python 08_ensemble_learning/gradient_boosting_regression.py
python 09_gaussian_naive_bayes_from_scratch/gaussian_naive_bayes.py
python 10_svm_from_scratch/svm.py
python 11_pca_from_scratch/pca.py
```

## 固定 Seed 的實驗結果

| 主線 | 選擇方式 | 本輪重現的結果 |
|---|---|---|
| KNN | Validation Accuracy 選 K=1 | Test ACC 0.9500 |
| K-Means | 比較 K=2／3／4，沒有分類 Test | Inertia 316.3220／214.3996／125.7815 |
| Linear Regression | Validation MSE 選 LR=0.01 | Test MSE 0.386479 |
| Logistic Regression | Validation BCE 選 LR=0.5 | Test BCE 0.092998；ACC 1.0000 |
| Decision Tree | Validation ACC 選 max_depth=10 | Test ACC 0.9750 |
| Random Forest | Validation ACC 選 10 棵樹 | Test ACC 0.9500 |
| L06 分類比較 | 四種策略的固定設定比較 | Validation：Single 0.925、Bagging 0.900、RF 0.975、AdaBoost 0.950；Test 未評估 |
| Gradient Boosting 回歸 | Validation MSE 選 100 輪 | Test MSE 0.394896 |
| XGBoost 示範 | 固定設定，非三套工具競賽 | Validation MSE 38.672060；Test 未評估 |
| Gaussian Naive Bayes | 固定模型設定 | Test ACC 0.9750 |
| Linear SVM | Validation ACC 選 C=10 | Test ACC 0.9250；最佳化未充分收斂驗證 |
| PCA | 至少保留 95% Train 變異，選最少維度 | 一維保留 97.90%；Test 還原 MSE 0.049783 |

大多數預測實驗為 200 筆、切分 120／40／40，XGBoost 是 300 筆、180／60／60；Seed 見各章 CONFIG。K-Means 是完整資料的無監督探索，NumPy 基礎章沒有模型評估。指標、任務與資料不同，不能跨表直接排名。

## 審查發現與已完成修正

- KNN 改成依當次 Validation 結果選 K，避免 Config 改變後仍寫死 K=1。
- NumPy／KNN／K-Means 入口隔離，匯入時不執行訓練或繪圖。
- Sigmoid 改成穩定等價計算，避免極端分數使 exp 溢位。
- 補上空森林、無效樹數、AdaBoost 輸入與票重配對、Naive Bayes 變異數下限、PCA 布林維度等防護。
- 補齊 NumPy 基礎章的圖檔保存，取消樹／森林執行時的阻塞式視窗。
- 核心依賴與選用 XGBoost 分開，備份、環境、快取及舊版中間圖不納入 Git。

## 尚未完成與實作限制

- Derivative、Chain Rule、Matrix Gradient、SVM 最佳化及 PCA 特徵分解的完整獨立推導仍待複習；各章圖解與回答修正均保留。
- SVM 是固定步幅批次次梯度教學版本，500 輪不是收斂保證。僅使用 Train 的額外比對顯示 C=0.1 尚能降低目標，不能將本表視為精確 SVM 解。
- 06 的樹對零改善切分會停止，無法學會需先零改善分割的 XOR；07／08 允許這種分割，屬有意保留的章節差異。
- 未實作 Kernel／RBF SVM、OOB、Soft Voting、Feature Importance 或完整 Boosting Framework。LightGBM／CatBoost 沒有本次訓練結果。
- L06 分類比較沒有完全配對的 Bootstrap 隨機序列，AdaBoost 基模型也較淺，不能據此單次結果證明因果優劣或穩定性。
- 單次合成資料與小型 Test 不構成真實應用效能保證；未完成重複切分、Calibration、公平性或部署評估。教學 API 仍有有限的輸入與數值範圍。
- PCA 保留變異不等於保留分類準確率；本次只檢查還原誤差，未評估下游模型。

## 資料、授權與發布狀態

資料均由程式生成；沒有真實設備或客戶資料、模型權重、套件原始碼或虛擬環境隨附。直接依賴為 NumPy（BSD 授權及其組件 notices）、Matplotlib（Matplotlib License Agreement）與選用 XGBoost（Apache-2.0）；依賴由套件管理器安裝，不隨附套件原始碼。Boosting 章節筆記保留官方參考文件連結。

本專案尚未指定根目錄 LICENSE；不宣稱已授予開源再利用授權。第三方套件的授權不等於本專案自身的授權。

目前版本聚焦演算法主線、必要教學檔、圖表及數學筆記。
