# Treasury ETF Futures Pair Trading

国债 ETF 与期货配对交易研究，包含 OU 基线、Kalman 时变状态模型、Optuna 参数搜索，以及 Numba 数据解析和回测加速。

主文件为 `notebooks/etf_futures_pair_trading.ipynb`。数据预处理 Notebook、早期 OU Python 实现及 ETF 申赎对照实验单独保留，避免混淆其用途。`results/pair_trading` 与 `results/ou_baseline` 保存本地已有的指标、参数搜索结果、敏感性分析和图表；原 Notebook 输出保留。

## 数据与处理

原实现清洗并对齐 ETF 与期货快照，处理期货换月、ETF 分红和连续价差。完整行情数据已作为 [Market data v1 Release 附件](https://github.com/WentaoZhu626/treasury-etf-futures-pairs/releases/tag/market-data-v1) 发布：32 个 CSV / XLSX 文件，原始约 1.8 GiB，压缩包约 208 MiB。包括国债 ETF、债券和 TL 主力期货快照、合并后的数据，以及 PCF、现金差额与债券条款表。

```bash
# 下载压缩包，校验 SHA-256，再还原每个数据文件
python prepare_market_data.py
# 如果已经手动下载附件，可使用本地压缩包
python prepare_market_data.py --archive /path/to/treasury-etf-market-data-v1.tar.gz --no-download
```

数据默认解压到 `data/market_data/`，各 Notebook 已使用这个目录；也可设置环境变量 `MARKET_DATA_ROOT` 指向自己的数据目录。`data/market_data_manifest.json` 列出完整文件清单、大小和 SHA-256。还原脚本逐文件校验，并拒绝覆盖内容不同的已有文件。

公开数据包只有数据表，不包含测试题目录、题目文件、PDF、DOCX、TXT 说明、环境文件或凭据。运行前仍应核对交易成本、时间区间、数据频率和策略参数。

## 结果口径

结果对应历史研究运行。请从 Notebook 和 `results/pair_trading/kalman_vs_ou.csv` 等文件读取精确口径，区分样本内优化与样本外验证。发布过程中未执行新的回测，也未重算性能加速比。

安装 `requirements.txt` 后用 Jupyter 查看和运行。源代码及结果经作者明确指定发布，原公司测试题、环境文件和无关笔试资料不在仓库中。
