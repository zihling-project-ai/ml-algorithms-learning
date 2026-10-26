# Review Fixes

2026-10-26：首次整理為本機可提交的 Educational／From-Scratch Repository。

- 核心數字／邊界行為檢查由 unittest 執行，預設範例由臨時副本執行。
- 舊範例入口加 main guard；KNN 依當次 Validation 動態選 K。
- 穩定 Sigmoid、空森林／票重配對／variance_floor／PCA 維度等必要防護。
- 分離核心與選用工具依賴，排除環境、備份、快取及舊中間圖。
- 完整發現、保留限制及流程審查見 [PUBLICATION_REVIEW.md](PUBLICATION_REVIEW.md)。

本輪沒有公開推送或大幅改寫教學結構。
