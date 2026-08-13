# ai-quant-system## 设计理念

> **AI量化的核心竞争力不是"用多复杂的模型"，而是"把不同模型放在对的位置上"。**

- **树模型(LightGBM)** 负责"挖Alpha"
- **强化学习(PPO)** 负责"调仓位"
- **多智能体(TradingAgents)** 负责"读信息"
- 各司其职、层层过滤

---

## 系统架构 (四层架构)

```
┌─────────────────────────────────────────────────────────────┐
│                     QuantPipeline (主调度器)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  第1层:初筛层 │───>│  第2层:精研层 │───>│  第3层:执行层 │  │
│  │ Screening    │    │ Research     │    │ Execution    │  │
│  │ LightGBM     │    │ TradingAgents│    │ PPO          │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         ↓                      ↓                  ↓         │
│    "挖Alpha"               "读信息"            "调仓位"      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              第4层: 风控层 (Risk Layer)              │  │
│  │         DoubleEnsemble 低过拟合风控                   │  │
│  │         个股风控 + 组合风控 + 市场风控                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 第1层: 初筛层 (Screening Layer)
- **模型**: LightGBM
- **功能**: 全市场因子选股，产出候选股票池
- **流程**:
  1. 获取股票池 (沪深300/中证500/全A股)
  2. 计算因子 (技术因子 + 量价因子 + 动量/反转因子)
  3. LightGBM训练 (横截面二分类: 未来5日涨跌)
  4. 预测打分 → 排序选股
  5. 行业分散约束

### 第2层: 精研层 (Research Layer)
- **模型**: TradingAgents 多智能体系统
- **功能**: 读研报/新闻做逻辑验真，排除风险标的
- **Agent**:
  - **SentimentAgent**: 情感分析 (关键词匹配 + 强度加权)
  - **RiskAgent**: 风险识别 (ST/质押/监管/流动性)
  - **NewsAgent**: 新闻分析 (时效性衰减 + 情感评分)

### 第3层: 执行层 (Execution Layer)
- **模型**: PPO (Proximal Policy Optimization)
- **功能**: 动态仓位微调，适应市场变化
- **组件**:
  - **PortfolioEnv**: 强化学习环境（已升级为标准 `gymnasium.Env` 接口）
  - **PPO**: 仓位优化策略
  - **自动加载**: `ExecutionLayer` 会尝试自动加载已保存的 PPO 模型
  - 回退: 等权分配 / 分数加权

### 第4层: 风控层 (Risk Layer)
- **模型**: DoubleEnsemble (双重集成学习)
- **功能**: 低过拟合设计做安全垫
- **层级**:
  - **个股风控**: 止损/移动止损
  - **组合风控**: VaR/波动率/集中度
  - **市场风控**: 熔断/波动率目标

---

## 项目结构

```
ai-quant-system/
├── config/
│   └── settings.yaml          # 系统配置文件
├── data/
│   ├── __init__.py
│   ├── fetcher.py             # 数据获取 (AKShare/yfinance/CSV)
│   └── processor.py           # 因子处理器 (50+因子)
├── layers/
│   ├── __init__.py
│   ├── screening.py           # 初筛层 (LightGBM)
│   ├── research.py            # 精研层 (TradingAgents)
│   ├── execution.py           # 执行层 (PPO)
│   └── risk.py                # 风控层 (DoubleEnsemble)
├── utils/
│   ├── __init__.py
│   ├── logger.py              # 日志系统
│   ├── config_loader.py       # 配置加载器
│   ├── data_helper.py         # 数据辅助函数
│   └── validator.py           # 数据验证
├── pipeline.py                # 主调度器
├── main.py                    # 主入口
├── requirements.txt           # 依赖清单
└── README.md                  # 本文档
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

```bash
python main.py --mode demo
```

### 3. 运行初筛

```bash
python main.py --mode screening --date 2026-07-21 --top-k 50
```

### 3.1 训练 Screening 模型

初筛层 (LightGBM) 支持两种训练数据源：

1. **面板模式（默认，离线）** —— 从已准备的 CSI300 日线面板训练，无需联网：

   ```bash
   # 默认读取 data/csi300_daily_2010_2026.csv.gz（约 20MB，需从项目网盘下载，见 data/README.md）
   python scripts/train_screening.py
   # 没有完整数据集时，自动回退到提交的小样本做冒烟测试
   python scripts/train_screening.py --panel data/csi300_sample.csv
   ```

   关键参数：`--lookback-days 0`（默认，使用全部历史）、`--min-cv-auc`（CV AUC 门槛，演示默认 0.0，生产建议 ≥0.60）。

2. **实时模式** —— 通过 `DataFetcher`（需网络 / AKShare）拉取数据并跑完整 `run_screening`：

   ```bash
   python scripts/train_screening.py --live --date 2026-07-21 --universe hs300
   ```

训练产物（模型元数据 / CV 报告）写入 `models/`；大体积权重 `screening_model.txt` 不纳入版本库。

### 4. 运行回测

```bash
python main.py --mode backtest --start 2024-01-01 --end 2025-12-31
```

### 5. 运行实盘

```bash
python main.py --mode live --max-days 10
```

### 6. 训练与部署 PPO

```bash
python scripts/train_ppo.py --episodes 50 --universe hs300
python scripts/deploy_ppo.py --input scripts/example_verified.json
```

---

## 配置说明

配置文件: `config/settings.yaml`

### 初筛层配置
```yaml
screening:
  enabled: true
  universe: "hs300"           # 股票池
  selection:
    top_k: 50                 # 选前50只
    min_score: 0.3            # 最低分
    industry_diversify: true  # 行业分散
  lightgbm:
    learning_rate: 0.05
    n_estimators: 500
    max_depth: 6
```

### 精研层配置
```yaml
research:
  enabled: true
  report_analysis:
    min_sentiment_score: 0.2
    keywords:
      positive: ["增长", "突破", "利好"]
      negative: ["亏损", "退市", "处罚"]
  exclusion:
    st_stocks: true
    high_pledged: true
    regulatory: true
```

### 执行层配置
```yaml
execution:
  enabled: true
  auto_load_model: true
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
  position:
    max_positions: 20
    min_position: 0.02
    max_position: 0.10
```

### 风控层配置
```yaml
risk:
  enabled: true
  metrics:
    max_drawdown: 0.15
    daily_loss_limit: 0.03
    var_confidence: 0.95
  position_risk:
    stop_loss: -0.08
    trailing_stop: -0.05
```

---

## 因子列表 (50+)

### 技术指标因子 (16个)
| 因子 | 说明 |
|------|------|
| ma_5 ~ ma_250 | 5/10/20/60/120/250日均线 |
| rsi_6/14/24 | 6/14/24日RSI |
| macd_hist | MACD直方图 |
| boll_position | 布林带位置 |
| atr_14 | 14日ATR |
| volume_ratio | 量比 |
| price_ma_* | 价格/均线比值 |
| volatility_* | 滚动波动率 |

### 量价因子 (16个)
| 因子 | 说明 |
|------|------|
| amihud | 流动性指标 |
| realized_volatility_* | 实现波动率 |
| skewness/kurtosis | 偏度/峰度 |
| momentum_* | 动量因子 |
| reversal_* | 反转因子 |
| volume_price_trend | 量价趋势 |
| obv | OBV指标 |

### 综合评分因子 (回退方案)
- 20日动量
- 5日动量
- 波动率
- 量比
- 均线位置
- RSI

---

## 绩效指标

系统自动计算以下指标:

| 指标 | 公式 | 说明 |
|------|------|------|
| Total Return | V_final/V_0 - 1 | 总收益率 |
| Annualized Return | (1+R)^(252/n) - 1 | 年化收益 |
| Sharpe Ratio | (R-rf)/σ × √252 | 夏普比率 |
| Max Drawdown | min((V-cummax)/cummax) | 最大回撤 |
| Win Rate | wins/total | 胜率 |
| Profit/Loss Ratio | avg_win/avg_loss | 盈亏比 |
| Calmar Ratio | ann_return/max_dd | Calmar比率 |
| Sortino Ratio | (R-rf)/downside_std × √252 | Sortino比率 |

---

## 数据源

| 数据源 | 类型 | 说明 |
|--------|------|------|
| AKShare | A股行情/基本面 | 免费，需API |
| yfinance | 港股/美股 | 免费 |
| CSV | 本地数据 | 自定义 |

---

## 依赖

### 核心依赖
- lightgbm >= 4.0
- scikit-learn >= 1.0
- pandas >= 1.5
- numpy >= 1.21
- akshare >= 1.0
- loguru >= 0.6

### 可选依赖
- stable-baselines3 >= 2.0 (PPO)
- nltk (NLP)

---

## 注意事项

1. **本系统仅供学习和研究使用**
2. 实盘交易需要额外配置交易接口
3. 回测结果不代表未来表现
4. 请根据自身风险承受能力决策

---

## License

MIT License
