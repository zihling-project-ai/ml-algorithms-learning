# L06｜Ensemble Learning 理論驗收與問答

## 驗收摘要

| 題目 | A1 評估 | 狀態 |
|---|---|---|
| Q1｜Bagging vs. Random Forest | 理解兩種隨機化機制，用詞需精確 | PASS |
| Q2｜AdaBoost vs. Gradient Boosting | 能區分 Sample Weight 與 Residual | PASS |
| Q3｜工程選型 | 混淆相同參數與公平比較，已補充教學 | 待補強 |

## Q1｜Bagging vs. Random Forest

**題目：**Bagging 與 Random Forest 都使用 10 棵 Decision Trees，為什麼森林還需要 Random Feature Selection？與只有 Bootstrap 的 Bagging 有何差別？

**原回答：**兩者都有重新抽樣並組成最終決策的功能，最大差異是 Random Forest 還會只選特定特徵；曾描述為「把不同方法的模型組成決策」。

**評估：PASS，補正兩個用詞。**

1. 本次 Bagging 使用相同類型的 Base Learner，也就是 Decision Tree，在不同 Bootstrap Samples 上訓練；不是組合不同演算法。
2. Random Forest 的 `max_features` 限制的是**每個節點**隨機考慮的候選 Features，不是整棵樹永遠固定使用某幾欄。

| 比較 | 本次 Bagging Trees | Random Forest |
|---|---|---|
| Bootstrap Sampling | 使用 | 使用 |
| 每個節點隨機選候選欄位 | 不使用，搜尋全部欄位 | 使用 |
| Base Learner | Decision Tree | Decision Tree |
| 主要目的 | 利用不同樣本組合與整合降低 Variance | 進一步降低各樹預測的相關性 |

**修正後摘要：**Bagging 利用 Bootstrap 讓各樹看到不同 Train 樣本組合；Random Forest 再於每個節點隨機限制候選欄位，增加多樣性、降低樹之間的相關性，讓集成更有機會降低整體預測變異。不保證每次準確率都提升。

## Q2｜AdaBoost vs. Gradient Boosting

**題目：**第一輪五筆資料有兩筆答錯。AdaBoost 下一輪如何處理？本次 Gradient Boosting Regression 的下一個 Learner 學什麼？

**原回答：**AdaBoost 放大錯誤答案的權重，提示下一輪注意；Gradient Boosting Regression 計算殘差，下一輪根據這個方向修正。

**評估：PASS。** 精確說法是提高**被錯誤分類樣本的訓練權重**，不是修改答案，或提高錯誤答案本身的權重。

| 比較 | 本次 AdaBoost 分類 | 本次平方誤差 Gradient Boosting 回歸 |
|---|---|---|
| 下一輪的改變 | 調整 Sample Weights | 重新建立 Residual Targets |
| 學習目標 | 原本的 0／1 標籤 | 真實值－目前整體預測 |
| 後續 Learner 的工作 | 更重視目前加權後的樣本 | 擬合目前剩餘誤差 |
| 最後整合 | 依 alpha 加權投票 | 起始預測＋各輪縮減後的修正 |

```text
AdaBoost：標籤不換，樣本重要性改變。
Gradient Boosting 回歸：輸入仍是 x，學習目標換成目前殘差。
```

殘差是訓練目標，不是最後輸出。Gradient Boosting 也能用於分類或其他 Loss，不能把所有情況都直接寫成原始標籤減預測值。AdaBoost alpha 的完整數學推導本輪不再加深。

## Q3｜工程選型：如何公平比較？

**題目：**廣告銷售資料包含預算、折扣、品牌。如何比較 Random Forest、XGBoost、LightGBM，選擇合適模型？

**原回答：**固定樹深度與切分次數，希望控制變因；認為深度到一定程度可能趨近一致，若仍有顯著差異，可能表示某方法更合適。

**評估：有控制變因的方向，但不能把相同超參數視為公平比較。**

### 為什麼相同深度不夠？

```text
Random Forest：max_depth＝5
XGBoost：max_depth＝5
LightGBM：max_depth＝5
```

這不代表三者具有相同容量、複雜度或運算成本：

- Random Forest 的樹分別訓練，再整合預測。
- XGBoost 的樹逐輪加入，改善整體 Loss，還涉及 Learning Rate、正則化等設定。
- LightGBM 採 Leaf-wise 生長，需同時考慮葉節點數、每葉樣本數等限制，不能只看深度。
- **深度足夠就會產生一致預測，沒有一般性的保證。** 即使都能貼合 Train，對未見資料仍可能不同。

固定相同參數可以用於特定機制的控制實驗，但不等於充分評估各模型對業務任務的適用性。

### 公平比較需要控制與記錄的內容

| 項目 | 合理做法 |
|---|---|
| Dataset | 相同資料版本與預測任務 |
| Train／Validation／Test | 相同切分，Test 不參與開發選模；時間資料需遵守先後關係 |
| Features | 相同、在預測當下可取得的輸入資訊 |
| Preprocessing | 避免 Leakage，轉換只在允許的訓練資料上擬合 |
| Evaluation Metric | 使用相同且符合業務成本的指標 |
| Hyperparameter Tuning | 給予合理、可比較的搜尋預算，而非強迫參數值完全相同 |
| Randomness | 固定或記錄 Seeds，必要時重複實驗 |
| Resource Cost | 記錄訓練時間、推論延遲與記憶體；測量條件一致 |
| Final Selection | 依 Validation 與業務限制選模，最後才使用 Test |

不同工具可以採用適合自己的編碼與內部最佳化，不必每項處理完全一樣；但不能讓某模型額外取得 Test Labels、未來資訊或其他模型沒有的業務資訊。

## 廣告案例：選模不只比較誤差

以下是 **A1 的教學示意數據，不是專案實測結果**。

| 模型 | Validation RMSE | 訓練時間 | 推論延遲 |
|---|---:|---:|---:|
| Random Forest | 7.2 件 | 3 秒 | 12 ms |
| XGBoost | 6.5 件 | 6 秒 | 4 ms |
| LightGBM | 6.7 件 | 2 秒 | 3 ms |

- 若重點是最低預測誤差，這組示例中的 XGBoost 是候選。
- 若需要頻繁重訓、計算資源有限，LightGBM 的效率可能更符合需求。
- 差距很小時，還需看重複實驗的不確定性、維護成本及部署限制，不能只用單次排名下結論。

實際比較前須先定義錯誤成本：MSE／RMSE 對大誤差較敏感；MAE 按絕對誤差計算。指標應反映業務需求，而不是看到哪個分數漂亮才選哪個。

**本專案實測範圍：**廣告案例目前只訓練 XGBoost，Validation MSE＝38.6721、RMSE 約 6.22 件；沒有 Random Forest 與 LightGBM 在這份廣告資料上的對照成績。上述示意表不可併入實驗證據。

## 待確認與複習

- Q1、Q2 已由 A1 評估通過。
- Q3 已說明公平比較原則，尚無後續重新回答或最終確認紀錄。
- 尚未完成三模型的受控調參與資源成本比較，不為補筆記而擴增實驗。
- Variance 的跨訓練資料重複驗證仍未實作。
- 本紀錄不改變既有 Test 的使用狀態，也不自行宣告 L06 正式驗收通過。

複習核心：**公平比較是相同任務、資料切分、可取得資訊、評估規則與合理搜尋預算，不是所有模型填入相同參數值。**
