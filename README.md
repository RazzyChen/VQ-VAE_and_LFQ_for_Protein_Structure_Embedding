# VQTokenizer

## 简介
VQTokenizer 是一个用于向量量化（Vector Quantization, VQ）和无查找量化（Lookup-free Quantization, LFQ）的深度学习工具包。该项目支持高效的数据分块、编码、训练和推理流程，适用于蛋白质结构嵌入任务。

## 主要功能
- 支持 VQ 和 LFQ 两种向量量化方法
- 灵活的 YAML 配置文件管理训练参数
- 包含数据预处理、分块、训练、推理等完整流程
- 可扩展的神经网络模块和数据管道
- 工具脚本支持数据集预处理、配置生成等

## Lookup-Free Quantization Layer (LFQ)
- 参考论文1: https://arxiv.org/abs/2410.13782
- 参考论文2：https://arxiv.org/abs/2310.05737

## 目录结构
- `infer.py`：推理脚本
- `VQ_trainer.py` / `LFQ_trainer.py`：训练脚本
- `config/`：训练配置文件
- `tools/`：数据预处理和工具脚本
- `vqtokenizer/`：核心代码（数据集、神经网络、工具函数等）

## 安装
推荐使用 Poetry 进行依赖管理：

```bash
poetry install
```

或使用 pip 安装依赖：

```bash
pip install -e .
```

## 用法示例
1. 配置训练参数（见 `config/train.yml` 或 `config/lfq_train.yml`）
2. 运行训练脚本：

```bash
python VQ_trainer.py --config config/train.yml
# 或
python LFQ_trainer.py --config config/lfq_train.yml
```

3. 推理：

```bash
python infer.py --config config/train.yml
```

## 工具脚本
- `tools/create_default_train_config.py`：生成默认训练配置
- `tools/precompute_patches_lmdb.py`：预处理数据集
- `tools/uncompress_pdb.sh`：解压数据集

## 许可证
本项目遵循 MIT 许可证，详见 LICENSE 文件。
