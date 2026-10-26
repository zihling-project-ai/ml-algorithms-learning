# ML Algorithms Learning

Educational Machine Learning implementations with Python, NumPy and Matplotlib. 
由 NumPy 實作，未使用 Scikit-learn 模型或切分工具。
期許藉由專案學習古典模型演算法：涵蓋資料建立、演算法核心、實驗比較、未見資料評估及數學圖解。


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

各章的 `*_learning.py` 保存小型推導範例，主程式聚焦實驗；
THEORY_QA.md 記錄回答與修正，不把所有學習歷程當成正式驗證證據。


## 一、主要學習演算法總表

| 演算法 | 學習類型／方法 | 核心原理 | 主要用途 |
|---|---|---|---|
| KNN | Supervised | 找最接近新資料的 K 個訓練樣本，根據鄰居預測 | Classification／Regression |
| K-Means | Unsupervised | 反覆分配樣本並更新群中心，使群內平方距離縮小 | Clustering |
| Linear Regression | Supervised | 學習 Features 與連續目標值之間的線性關係 | Regression |
| Logistic Regression | Supervised | 將線性分數透過 Sigmoid 轉成類別 1 的預測機率 | Binary Classification |
| Decision Tree | Supervised | 利用 Feature Threshold 逐步切分資料 | Classification／Regression |
| Random Forest | Supervised；Ensemble／Bagging | 多棵隨機化 Decision Trees 整合預測 | Classification／Regression |
| AdaBoost | Supervised；Ensemble／Boosting | 本次二元分類版本逐輪提高被錯誤分類樣本的相對訓練權重 | 本次實作為 Classification |
| Gradient Boosting | Supervised；Ensemble／Boosting | 逐輪加入 Learner，沿降低 Loss 的方向修正預測 | Classification／Regression |
| Naive Bayes | Supervised；Probabilistic | 使用 Bayes Theorem，假設 Features 在給定 Class 後條件獨立 | Classification |
| SVM | Supervised | 分類時尋找兼顧 Margin 與違規懲罰的邊界 | Classification；回歸另有 SVR |
| PCA | Unsupervised | 尋找投影變異最大的正交主方向 | Dimensionality Reduction |



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

大多數預測實驗為 200 筆、切分 120／40／40，XGBoost 是 300 筆、180／60／60；Seed 見各章 CONFIG。
K-Means 是完整資料的無監督探索，NumPy 基礎章沒有模型評估。指標、任務與資料不同。



## 尚未完成與實作限制

- Derivative、Chain Rule、Matrix Gradient、SVM 最佳化及 PCA 特徵分解的完整獨立推導仍待複習；各章圖解與回答修正均保留。
- SVM 是固定步幅批次次梯度教學版本，500 輪不是收斂保證。僅使用 Train 的額外比對顯示 C=0.1 尚能降低目標，不能將本表視為精確 SVM 解。
- 06 的樹對零改善切分會停止，無法學會需先零改善分割的 XOR；07／08 允許這種分割，屬有意保留的章節差異。
- 未實作 Kernel／RBF SVM、OOB、Soft Voting、Feature Importance 或完整 Boosting Framework。LightGBM／CatBoost 沒有本次訓練結果。
- L06 分類比較沒有完全配對的 Bootstrap 隨機序列，AdaBoost 基模型也較淺，不能據此單次結果證明因果優劣或穩定性。
- 單次合成資料與小型 Test 不構成真實應用效能保證；未完成重複切分、Calibration、公平性或部署評估。教學 API 仍有有限的輸入與數值範圍。
- PCA 保留變異不等於保留分類準確率；本次只檢查還原誤差，未評估下游模型。


## 資料、授權與發布狀態
資料均由程式生成；沒有真實設備或客戶資料、模型權重、套件原始碼或虛擬環境隨附。
直接依賴為 NumPy（BSD 授權及其組件 notices）、Matplotlib（Matplotlib License Agreement）與選用 XGBoost（Apache-2.0）；
依賴由套件管理器安裝，不隨附套件原始碼。Boosting 章節筆記保留官方參考文件連結。
本專案尚未指定根目錄 LICENSE；不宣稱已授予開源再利用授權。第三方套件的授權不等於本專案自身的授權。
目前版本聚焦演算法主線、必要教學檔、圖表及數學筆記。
