# Exploration Mode統計データ

このディレクトリには、MarioタスクでのExploration Mode（exploration-mode-slider）パラメーターの使用統計データが含まれています。

## ファイル一覧

- `exploration_mode_statistics.py` - 統計分析スクリプト
- `exploration_mode_statistics.pdf` - 可視化結果（4つのグラフ）
- `exploration_mode_timeline.csv` - タイムスタンプ付きの変更履歴
- `exploration_mode_summary.csv` - 集計統計データ
- `exploration_mode_by_user.csv` - ユーザー別の使用統計

## 統計結果の概要

- **スタイル変更回数**: 合計389回
  - bold: 216回 (55.5%)
  - cautious: 109回 (28.0%)
  - standard: 64回 (16.5%)

- **使用可能な手法**: glv_bo_hybridのみ（bo、cma_es、manualでは使用不可）

- **フェーズ別使用状況**: practiceとmainフェーズで同様の傾向

## 使用方法

統計データを再生成する場合：

```bash
python3 exploration_mode_statistics.py
```

## 注意事項

- 現在のログファイルには**designタスクのログは含まれていません**（すべてmarioタスクのみ）
- ログファイルは `/home/takahito/Develop/Hummingbird-Kernel_a/app/results/log_*.jsonl` から読み込まれます
