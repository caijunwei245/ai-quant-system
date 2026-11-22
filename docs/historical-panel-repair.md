# Historical CSI300 panel repair

## Source finding

The uploaded `data0727/csi300_daily_2010_2026.csv.gz` panel uses AKShare/Eastmoney forward-adjusted (qfq) OHLC values. The independent `stock_zh_a_hist_em` comparison for `600000` reproduced the panel's negative qfq prices exactly, while the unadjusted and hfq prices remained positive.

An independent AKShare/Sina `stock_zh_a_hist_sina(adjust="qfq")` comparison was then performed on eight representative stocks spanning the affected years. The returned qfq OHLC prices were positive and internally consistent. Every panel date missing from the Sina series had zero panel volume; no traded panel date was missing. Sina volume was approximately 100 times the panel volume (shares versus lots), and traded-row amount relative error was below 0.03% at the 99th percentile in the sample.

## Repair procedure

The repair script `scripts/repair_historical_panel_sina.py`:

1. Fetches Sina qfq daily bars for every stock code in the panel.
2. Requires all traded panel dates to have a matching Sina bar.
3. Checks positive, internally consistent OHLC values.
4. Reconciles the Sina-to-panel volume unit ratio and traded-row amount values.
5. Replaces only `open/high/low/close`; preserves the original panel's dates, codes, names, volume, amount, turnover and other trading fields.
6. For a missing Sina date, permits a carry-forward close only when the original panel volume is zero. The imputed no-trade rows are explicitly marked in `price_source`.
7. Fails closed if any stock cannot be fully reconciled.

The repaired output is a new file, `data/csi300_daily_2010_2026_repaired.csv.gz`. The original uploaded file is retained unchanged.

## Scope and limitations

This repair addresses the price-adjustment defect only. It does not fix survivorship bias from using a fixed 300-stock universe, historical index membership, point-in-time fundamentals, corporate-action source completeness, or other factor-proxy issues. A successful OHLC repair is not evidence of alpha or deployable trading performance.
