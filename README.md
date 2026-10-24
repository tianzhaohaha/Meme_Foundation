# TumEmo 图文情绪分类：六种训练策略与Qwen零样本评估

使用 **ViT-B/16 + RoBERTa-base**，对比直接训练、源二分类迁移和融合层消融六种方案。训练、准确率和效率指标统一记录到 W&B，支持SSH后台运行和checkpoint恢复。

另在同一2,000条validation上完成Qwen3-VL-4B/8B-Instruct零样本七分类，accuracy分别为 **47.75% / 48.90%**，macro-F1为 **0.4538 / 0.4728**。原V1/V4在validation上为57.55%/56.50%；这些数字与下方test结果分开比较。详见[技术报告](reports/experiment_review.md#qwen零样本七分类同一2000条validation)和[VLM复现说明](docs/vlm_validation.md)。

| 方案 | 数据与训练过程 | 七分类阶段更新范围 |
|---|---|---|
| V1 Direct | 20,000 + 5,000条，直接七分类 | ViT、RoBERTa、融合层、分类头 |
| V2 Full | 20,000条二分类 → 5,000条七分类 | 全参数 |
| V3 Frozen | 共享V2的二分类checkpoint → 5,000条七分类 | 仅新分类头，1,799个参数 |
| V4 Direct 5k | 仅固定微调清单中的5,000条，直接七分类 | 全参数；从通用ViT/RoBERTa权重开始，不做二分类预训练 |
| V5 Reset fusion | 共享源二分类编码器 → 重新初始化融合层及七分类头 → 5k七分类 | 全参数；与V2相比只改变融合层初始化 |
| V6 Tune fusion | 共享源二分类编码器和融合层 → 新七分类头 → 5k七分类 | 冻结两个编码器，训练融合层及头；395,783个参数 |

本仓库保存源代码、固定样本清单、测试和已核验的聚合实验报告。数据ZIP、图片/文本、模型权重、环境缓存、运行日志和凭据均不在Git中。

**数据获取说明：** 本次使用的 `subset_50000.zip` 暂无公开下载地址，需要向仓库维护者获取相同数据包。仅克隆本仓库即可阅读报告、检查代码和重绘已有结果；完整训练还需要数据包、预训练模型及GPU。本仓库不分发任何实验checkpoint，迁移实验所需的二分类源模型会在首次运行时自动训练。

## 从 GitHub 开始

```bash
git clone https://github.com/tianzhaohaha/Meme_Foundation.git
cd Meme_Foundation
```

只查看和重绘已发布结果，无需数据、GPU或W&B账号：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python tools/plot_results.py
python tools/plot_vlm_results.py
```

完整训练使用下面的环境和数据准备步骤，或者直接按 [复现指南](docs/reproduction.md) 执行。报告中的历史数值来自已完成的seed42实验；读者重跑的输出写入新的运行目录，不会改写已提交的报告。

## 已完成的实验

seed42，NVIDIA H200 NVL，batch16，FP32；七个监督训练W&B run均已完成，云端summary与逐epoch曲线已和本地记录核对。

| 方案 | 测试accuracy | 测试macro-F1 | 完整方案阶段耗时 | 七分类训练轮数 |
|---|---:|---:|---:|---:|
| V1 Direct | **57.65%** | **0.5641** | 17.74分钟 | 6，最佳第2轮 |
| V2 Full | 54.30% | 0.5162 | 21.00分钟 | 6，最佳第2轮 |
| V3 Frozen | 42.10% | 0.2633 | 23.22分钟 | 20，最佳第18轮 |
| V4 Direct 5k | 56.90% | 0.5484 | **8.63分钟** | 12，最佳第8轮 |
| V5 Reset fusion | 54.85% | 0.5283 | 22.41分钟 | 8，最佳第4轮 |
| V6 Tune fusion | 54.75% | 0.5276 | 19.34分钟 | 8，最佳第4轮 |

完整方案耗时包含设置、训练、验证、保存和最终测试；V2/V3/V5/V6均计入共享二分类的原始成本。V3冻结版本每步快约2.54倍、第二阶段峰值显存少约81.4%，但训练轮数更多，第二阶段总耗时反而增加约50%。V4完整方案耗时8.63分钟，与V2相比少了源二分类预训练；V2测试accuracy相对V4变化-2.60个百分点。V1使用更多七分类标签，不能把它与其他版本的差异完全归因于训练策略。

![准确率、耗时与收敛](reports/figures/comparison.png)

- [完整结果分析与W&B链接](reports/experiment_review.md)
- [checkpoint逐文件对应表](reports/checkpoints.md) / [CSV](reports/checkpoints.csv)
- [可重绘的聚合数据](reports/data/seed42.json)
- [实验配置与比较口径](docs/experiment_protocol.md)
- [Qwen零样本validation协议及运行命令](docs/vlm_validation.md)
- [Binary/V4/V5/V6的PCA与t-SNE可视化报告](reports/embedding_visualization.md)

## 目录

```text
Meme_Foundation/
  training/                 # 六个版本启动器，以及baseline/fusion/all后台队列
  embedding_analysis/       # Binary/V4/V5/V6特征提取和PCA/t-SNE可视化
  tests/                    # 小模型集成测试、后台队列测试
  tools/plot_results.py      # 从聚合结果重绘图表
  data/manifests/            # 20k/5k训练清单、验证/测试ID清单
  docs/                     # 实验协议、可重复运行说明
  reports/                  # 结果分析、checkpoint索引、图表与聚合指标
  requirements.txt          # 训练依赖；PyTorch单独安装
  requirements-analysis.txt # 可选绘图依赖
  requirements-reproduction.txt # 本次历史实验的PyTorch版本
  requirements-lock-cu128.txt # Linux / Python 3.12历史依赖版本锁定
  tools/verify_data.py        # 验证数据包及不依赖机器路径的数据指纹
```

## 创建环境

推荐Linux、Python 3.12、NVIDIA GPU。以下命令均在仓库根目录运行。创建新环境示例：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 新环境示例：相互匹配的PyTorch / torchvision，CUDA 12.8 wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python training/check_environment.py
```

PyTorch和torchvision须匹配，CUDA wheel还需要相容的驱动；请结合本机硬件核对[官方版本对应表](https://github.com/pytorch/vision#installation)。本次报告的实测环境为Python 3.12.0、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128；上面的2.10环境示例尚未重跑完整实验，改变框架版本后应重新验证，不应期待逐位相同的结果。

如必须在独立环境中重建历史版本，可使用：

```bash
python -m pip install -r requirements-lock-cu128.txt
```

历史PyTorch 2.8在已知反序列化漏洞影响范围内；只加载可信来源的本地权重，受限加载不代表任意未知checkpoint安全。修复版本和边界见[官方公告](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)。新训练checkpoint采用受限加载兼容格式；旧可信pickle格式才需要显式 `--trust-legacy-checkpoint`。

## 准备数据和预训练模型

取得有使用权限的 `subset_50000.zip`，放到 `data/subset_50000.zip`。ZIP不需要手动解压，内部必须有：

```text
subset_50000/all_data_id_and_label_with_polarity.txt
subset_50000/all_data/<id>.jpg
subset_50000/all_data/<id>.txt
```

固定清单已经在 `data/manifests/` 中。默认完整使用20,000和5,000训练样本，从剩余样本抽取共同的2,000验证和2,000测试，排除与训练数据重复的图片和规范化非空文本。细节见 [data/README.md](data/README.md)。

```bash
bash training/run_v1_direct.sh --prepare-only
python tools/verify_data.py --data data/subset_50000.zip
```

校验脚本检查原始ZIP的SHA-256，以及29,000条准备后记录的ID、顺序、标签、图片哈希和清洗后文本指纹。记录中的绝对路径不参与该校验；不同机器可以得到相同的可移植数据指纹。参考值在 `data/dataset_signature.json`。

首次正式训练会下载ImageNet ViT权重及 `FacebookAI/roberta-base` 权重/分词器。RoBERTa固定到本机实验缓存记录的revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`，避免上游main变化。启动脚本默认把缓存放在仓库 `.cache/`；已有缓存可通过 `HF_HOME`、`TORCH_HOME` 指定。联网预先缓存后可设置 `HF_HUB_OFFLINE=1` 使用本地Hugging Face模型。

## W&B 登录与训练

前台完成登录，不把API key写入代码或命令文件：

```bash
.venv/bin/wandb login
export WANDB_PROJECT=tumemo-training-comparison
export WANDB_RUN_GROUP=three-strategies-seed42
# 如使用团队空间：export WANDB_ENTITY=你的团队名
```

SSH场景完整复现六个实验：

```bash
bash training/run_all_nohup.sh --suite all
bash training/run_all_nohup.sh status --suite all
tail -f training/background_all/queue.log
# 停止后，用同样的配置恢复
bash training/run_all_nohup.sh stop --suite all
bash training/run_all_nohup.sh --suite all -- --resume
```

`--suite all`按V1→V2→V3→V4→V5→V6串行执行，只训练一次共享源二分类模型。没有W&B账号时可在启动命令前加 `WANDB_MODE=offline`，仍保存本地全部指标；离线日志模式的耗时不应要求与原在线模式完全一致。

不指定 `--suite` 时保留原V1→V2→V3队列；下面的status/stop示例对应这个默认队列，运行all套件时应加 `--suite all`。

总脚本使用 `nohup + setsid`，默认GPU 0，各版本串行运行，避免争用GPU造成计时失真。迁移版本复用同一个源checkpoint；失败或正常停止后不会启动下一个版本。

```bash
# 状态与日志
bash training/run_all_nohup.sh status
tail -f training/background_comparison/queue.log
tail -f training/background_comparison/run_v1_direct.log

# 等待当前epoch保存后，停止整个队列
bash training/run_all_nohup.sh stop

# 用相同配置恢复；完成的阶段会跳过
bash training/run_all_nohup.sh -- --resume
```

也可以单独运行：

```bash
bash training/run_v1_direct.sh
bash training/run_v2_finetune.sh
bash training/run_v3_frozen.sh
bash training/run_v4_direct_5k.sh
bash training/run_v5_reset_fusion.sh
bash training/run_v6_tune_fusion.sh
```

补充V4时，单独后台运行，沿用前三版的输出根目录和数据缓存：

```bash
mkdir -p training/background_v4
nohup setsid bash training/run_v4_direct_5k.sh \
  --stop-file training/background_v4/STOP \
  > training/background_v4/train.log 2>&1 < /dev/null &
echo $! > training/background_v4/worker.pid
tail -f training/background_v4/train.log
```

V4写入 `training/runs_comparison/direct_5k/seed_42/`，不会覆盖V1。需要停止时运行 `touch training/background_v4/STOP`，等待当前轮保存；恢复前移除该STOP文件，并在相同启动命令中添加 `--resume`。使用 `--suite all` 时已包含V4，无需单独启动。为使计时可比，请避免同时运行多个队列。

V4仅将5,000条目标样本用于梯度更新。预处理仍读取20k/5k清单以复用原来的验证/测试划分，不会将20k样本加入V4训练。它没有第二阶段，因此 `comparison/stage2_*`为空；与V2/V3第二阶段比较时使用V4自身的 `efficiency/*`，完整方案比较使用共同的 `comparison/total_*`。

V5/V6融合层消融使用独立后台队列；已有源二分类结果会复用，保持原V4编号及所有历史目录：

```bash
bash training/run_fusion_nohup.sh
bash training/run_fusion_nohup.sh status
tail -f training/background_fusion/queue.log
tail -f training/background_fusion/run_v5_reset_fusion.log
tail -f training/background_fusion/run_v6_tune_fusion.log
bash training/run_fusion_nohup.sh stop
# 停止后的恢复，与启动时保持同样参数
bash training/run_fusion_nohup.sh -- --resume
```

该队列通过nohup + setsid按V5→V6串行运行。V5将源融合层排除在迁移加载之外，保留创建模型时的全新融合层，之后全部参数可更新。V6加载源融合层并训练它，只有两个编码器处于eval/no_grad。融合层位于no_grad范围之外，梯度可正常更新其权重。两者均使用与V2/V3相同的源checkpoint、5k清单、验证/测试集、优化器和早停规则；源预训练成本计入完整方案耗时，第二阶段成本单独记录。

自定义GPU、数据路径和输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash training/run_all_nohup.sh \
  --log-dir /absolute/path/experiment_a_logs -- \
  --data /absolute/path/subset_50000.zip \
  --output /absolute/path/experiment_a \
  --batch-size 16 --epochs 20 --binary-epochs 20 --seed 42
```

`--log-dir`是总脚本参数，后面的 `--`分隔训练参数。自定义日志目录后，status/stop/恢复也须使用相同日志目录。恢复时保持数据、输出目录和超参数一致；改变配置使用新输出目录。单阶段手工创建的 `STOP` 文件需自行删除；`--resume`只自动清理总队列自己的STOP。

默认参数：Adam，编码器LR `5e-6`，融合层/分类头LR `1e-4`，weight decay `0.01`，dropout `0.5`，5% warmup + cosine，batch16，FP32，文本最多96 tokens，图片224×224。每阶段最多20轮、连续4轮无提升早停；二分类按验证macro-F1选模，七分类按验证accuracy选模。可用 `bash training/run_v1_direct.sh --help` 查看所有选项。

无网络时显式使用离线记录，联网后同步每个阶段的离线目录：

```bash
WANDB_MODE=offline bash training/run_all_nohup.sh
.venv/bin/wandb sync training/runs_comparison/direct/seed_42/wandb/offline-run-*
```

对比实验保持同一日志模式。`--wandb-mode disabled`只写本地指标；在线登录/同步失败会明确报错，不会悄悄关闭W&B。

## 输出与checkpoint

```text
training/runs_comparison/
  direct/seed_42/
  full/seed_42/
  frozen/seed_42/
  direct_5k/seed_42/
  reset_fusion/seed_42/
  tune_fusion/seed_42/
  shared_binary/<配置指纹>/seed_42/
  comparison_seed_42.csv
  comparison_seed_42.json
```

每阶段包含 `best.pt`（按指定验证指标）、`best_accuracy.pt`（按验证accuracy）、`last.pt`（最后完整epoch，恢复用）、`config.json`、`history.json`、`efficiency.json`、`results.json`。六个七分类阶段另有 `test_metrics.json`。测试只在验证集选好模型后执行；最终报告使用 `best.pt`，不使用 `last.pt`。

二分类checkpoint用于初始化源特征，不能直接当七分类模型使用。完整checkpoint包含模型、优化器和随机状态，体积通常比纯模型权重大；不会上传到Git。已完成实验的逐文件索引见 [reports/checkpoints.md](reports/checkpoints.md)，这些历史文件仍位于原项目目录。

## 检查与重绘报告

```bash
# 小模型测试，不下载模型，也不进行正式GPU训练；W&B测试需要本地socket权限
python -m unittest discover -s tests -v

# 准备数据并缓存模型之后，两个真实样本的GPU检查
python -u training/smoke_comparison_gpu.py

# 可选：重绘本仓库聚合结果的图表
python -m pip install -r requirements-analysis.txt
python tools/plot_results.py
```

旧的单独二分类入口 `training/run_train.sh ZIP`仍保留，它使用原20k内部训练/验证划分，与六版本实验不同，不应混合比较。完整复现请使用 `run_all_nohup.sh --suite all`；不指定suite时仅运行V1→V2→V3，`run_fusion_nohup.sh`仅运行V5→V6。

## 发布范围

仓库只跟踪代码、依赖版本、样本ID/标签清单、数据指纹、测试及技术报告，另保存Qwen的逐样本标签预测以便核验指标。checkpoint索引是历史文件的元数据，不代表GitHub提供了对应权重；重新训练后会生成自己的checkpoint。

数据包目前需向仓库维护者获取。预训练模型由训练代码从其官方来源下载，数据和预训练模型遵循各自使用条件。本仓库暂未附通用开源许可证。
