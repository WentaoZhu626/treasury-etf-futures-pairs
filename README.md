# Treasury ETF Futures Pair Trading

国债 ETF 与期货配对交易研究，包含 OU 基线、Kalman 时变状态模型、Optuna 参数搜索，以及 Numba 数据解析和回测加速。

主文件为 `notebooks/etf_futures_pair_trading.ipynb`。数据预处理 Notebook、早期 OU Python 实现及 ETF 申赎对照实验单独保留，避免混淆其用途。`results/pair_trading` 与 `results/ou_baseline` 保存本地已有的指标、参数搜索结果、敏感性分析和图表；原 Notebook 输出保留。

## 数据与处理

原实现清洗并对齐 ETF 与期货快照，处理期货换月、ETF分红和连续价差。原始高频数据未加入普通 Git 仓库：本地部分单个 CSV 超过100 MiB，期货主力快照约952 MiB。这里保留代码、已有结果及小体积派生表。

运行前需自行准备原 Notebook 使用的 ETF、期货、分红和换月数据，调整数据路径，并检查交易成本、时间区间、数据频率与参数。仓库没有声称只凭附带的小表即可重现完整历史结果。

## 结果口径

结果对应历史研究运行。请从 Notebook 和 `results/pair_trading/kalman_vs_ou.csv` 等文件读取精确口径，区分样本内优化与样本外验证。发布过程中未执行新的回测，也未重算性能加速比。

安装 `requirements.txt` 后用 Jupyter 查看和运行。源代码及结果经作者明确指定发布，原公司测试题、环境文件和无关笔试资料不在仓库中。
