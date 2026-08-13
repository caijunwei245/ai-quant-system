# 数据目录 (Data)

## CSI300 日线面板 (csi300_daily_2010_2026.csv.gz)

初筛层 (ScreeningLayer) 的训练数据源：沪深300成分股的**日线长表面板**。

| 列名 | 含义 |
| --- | --- |
| `code` | 代码，形如 `sh600000` / `sz000001`（加载时自动归一化为 6 位纯数字） |
| `name` | 股票名称 |
| `date` | 交易日 (YYYY-MM-DD) |
| `open` / `high` / `low` / `close` | 开/高/低/收 |
| `volume` | 成交量（手） |
| `amount` | 成交额（元） |
| `turnover` | 换手率 |

时间跨度约 2010–2026，约 1.98 千万行、300 只股票。

### 获取完整数据集

完整文件约 20MB，未纳入 Git 仓库（受单文件体积与 GitHub API 限制）。
请从项目网盘 (Project Drive) 下载 `csi300_daily_2010_2026.csv.gz` 并放置到本目录：

```
data/csi300_daily_2010_2026.csv.gz
```

下载脚本示例见仓库 `data0727/`（`hs300_downloader.py`）。

### 已提交的小样本

`data/csi300_sample.csv`（约 12 只股票、最近 ~250 个交易日）已纳入仓库，
用于在没有完整数据集时跑通训练流水线的冒烟测试：

```bash
python scripts/train_screening.py --panel data/csi300_sample.csv
```

## 加载器

`panel_loader.py` 负责读取面板并转换为逐股票 DataFrame：

- `load_panel(path)` —— 读取 `.gz`/`.csv`，解析日期、归一化代码、清理 BOM
- `iter_stock_frames(df)` —— 逐股票产出 `(code, frame)`，仅保留历史 ≥ 60 日
- `build_stock_frame(df, code)` / `list_universe(df)` —— 单只抽取 / 股票池列表

训练脚本 `scripts/train_screening.py` 默认读取完整面板；
若缺失则自动回退到 `csi300_sample.csv` 做冒烟运行。
