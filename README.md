# BsplineKAN_LMM_V1

## 项目简介
本仓库包含基于 B 样条（B-spline）的 KAN 层实现及若干动力系统的实验 notebooks，用于对常见 ODE 系统进行数据驱动建模与对比。核心实现位于 `BKAN.py`，实验涵盖二维三次阻尼振子、Lorenz 系统与糖酵解振荡器等。`Cubic2D/` 目录保存了不同平滑/多步方法（AM1、AM4、RTS、Savitzky-Golay）下的结果数据与绘图脚本。

## 目录结构
- `BKAN.py`：B-spline KAN 线性层与多层封装（`SplineLinearLayer`、`DeepKAN`）。
- `RTS.py`：RTS（Rauch–Tung–Striebel）平滑器实现。
- `plotting.py`：Matplotlib LaTeX 风格绘图辅助函数。
- `Cubic2D.ipynb`：二维阻尼振子示例与训练/可视化流程。
- `Lorenz.ipynb`：Lorenz 系统示例与训练/可视化流程。
- `Glycolytic.ipynb`：糖酵解振荡器示例与训练/可视化流程。
- `odes.ipynb`：简单线性 ODE 示例。
- `test.ipynb`：临时实验/测试笔记。
- `Cubic2D/`：实验结果与不同平滑策略的数据与绘图 notebook。
- `MLNN-base/`：多步神经网络基线与参考实现（含独立 README）。

## 运行环境
- Python 3.8+（当前环境中可见 3.9 生成的缓存文件）
- 主要依赖：`torch`、`numpy`、`scipy`、`matplotlib`、`tqdm`
- 部分 notebook 依赖：`nodepy`、`pywt`、`jupyter`

## 使用方式
1. 安装依赖（示例）
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install torch numpy scipy matplotlib tqdm nodepy pywt jupyter
   ```
2. 运行 notebook
   ```bash
   jupyter lab
   ```
3. 打开对应的 `.ipynb` 完成数据生成、训练与可视化。

## 关键实现说明
- `SplineLinearLayer` 当前仅返回样条部分输出（`base_output` 已注释），若需要线性项叠加可在 `BKAN.py` 中启用。
- `DeepKAN.regularization_loss` 依赖 `SplineLinearLayer._regularization_loss`，当前未在 `BKAN.py` 中实现，若需正则项请补充对应函数。
- `_update_knots` 支持根据输入自适应更新样条结点，可在训练中按需启用。

## 数据与结果
- `Cubic2D/` 内含 `.npy` 结果文件与不同平滑方法的对比数据。
- `RTS 卡尔曼平滑/` 与 `Savitzky-Golay_51/` 子目录提供对应平滑策略的结果与绘图 notebook。
