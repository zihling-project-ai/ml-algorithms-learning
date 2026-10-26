# A1-P01-L08-A｜Linear SVM 理論驗收

本次以兩題確認 Margin 與 C 的核心理解，不要求重新推導最佳化公式。

**驗收結果：Q1 PASS；Q2 判斷正確，原因經補充說明，獨立解釋仍待確認。**

## Q1｜Margin

### 題目

為什麼 SVM 不只追求 Training Accuracy，還要考慮 Margin？Margin 較大的分類邊界，對新資料可能有什麼好處？

### 原始回答

> 因為很多的線性權重＋偏移值都能很好的分類，但若考量 Margin，可以在未見的新資料中增加誤判的寬容度，也就是說如果 Margin 抓取適當，新資料也不容易被誤判錯誤分類。

### 評估與修正

**PASS。** 已理解：Training Accuracy 相同的邊界，仍可能具有不同的間隔與穩健性。

SVM 在分類違規懲罰與較大 Margin 之間取捨。較大 Margin 提供對某些小幅輸入變化的緩衝，但不保證未知資料一定分類正確，仍取決於資料分布、尺度與模型假設。

Soft-Margin SVM 允許部分樣本落入間隔甚至分錯，因此不能把「Margin 大」解讀為所有 Training Samples 都離邊界更遠。這裡的容錯直覺是對輸入擾動的緩衝，不等於忽略分類錯誤。

## Q2｜C Parameter

### 題目

假設 C 從 0.1 增加到 100，模型對分類錯誤或 Margin Violation 的容忍程度通常會如何變化？這是否保證 Validation Accuracy 增加？

### 原始回答

> 變低，不保證 Validation Accuracy 增加，但解釋不出為什麼。

### 評估

**兩項判斷均正確，原因需要補強。** 尚未獨立連結 C、Margin、Overfitting 與 Generalization。

### 補充理解

本次訓練目標為：

```text
總目標 = 權重代價 + C × 平均 Hinge Loss
```

C 增加，代表分類錯誤與間隔不足的相對懲罰變重，模型通常更不願容忍這些違規。

假設 Training Set 含有少數 Noise 或 Outliers：

- C 較小：違規懲罰較輕，可能容許少數樣本違規，不必過度配合局部異常；過小也可能 Underfitting。
- C 較大：更重視降低 Training 違規程度，可能對雜訊較敏感，造成 Generalization 變差；但不必然 Overfitting。

| 設定方向 | 對違規的相對懲罰 | 可能情況 |
|---|---|---|
| C 較小 | 較輕 | 可能正則化過強、欠擬合 |
| C 較大 | 較重 | 可能過度配合雜訊，也可能改善欠擬合 |

「小」與「大」取決於資料尺度及目標定義，0.1、10、100 並不是普遍有效的欠擬合／適中／過擬合分界。

Training 違規程度降低，不代表未見資料的 Accuracy 一定提高；Hinge Loss 與 Accuracy 也不是同一個指標。因此不能直接選最大的 C，應依 Validation 表現選模。

### 核心記憶

> C 越大，越重視減少 Training Data 的 Margin Violations；但改善 Training 目標不保證改善 Generalization，所以 Validation Accuracy 不一定上升。

以上雜訊情境屬於概念說明，並非本次已執行的 Outlier 實驗，也未提供精確求解的邊界示意圖。

## 對照本次實驗

| C | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 0.1 | 0.5917 | 0.5750 |
| 1 | 0.9250 | 0.8500 |
| 10 | 0.9333 | 0.9000 |

依 Validation Accuracy 選定 C=10，Test Accuracy=0.925（37／40）。本次候選只有 0.1、1、10，**沒有測試 C=100**。10 恰好也是已測候選中的最大值，但選擇依據是 Validation 分數，不是 C 的大小。

這些結果來自固定學習率 0.05、500 次更新的次梯度實作，尚未完成充分收斂的驗證，不能據此宣稱已找到每個 C 的精確最佳解。

## 學習狀態與後續

| 項目 | 狀態 |
|---|---|
| Margin 的用途與輸入變動緩衝 | 核心理解 PASS |
| C 增大時，對違規的容忍度降低 | 判斷正確 |
| C 不保證提高 Validation Accuracy | 判斷正確，原因已補充；Recall 待確認 |
| NumPy 主線、Validation 選模與 Test | 已完成實作 |
| 邊界、Hinge 次梯度與更新公式的獨立推導 | 待複習 |
| 陌生情境的獨立應用 | 尚未驗收 |

本輪不追加數學推導或重做 SVM。以上紀錄區分已能回答、經說明理解與尚未獨立驗證的能力；後續接 L08-B｜PCA，再進行 Phase 01 整合驗收。

主線與實驗設定見 [README.md](README.md)，完整推導及教學圖片見 [SVM_DERIVATION_NOTES.md](SVM_DERIVATION_NOTES.md)。
