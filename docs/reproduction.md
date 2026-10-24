# 从下载仓库到复现实验

本仓库发布代码、样本划分清单、环境版本和完整分析材料，**不发布checkpoint或完整原始数据**；报告仅展示少量样本缩略图和文本节选。`subset_50000.zip` 目前没有公开下载地址，需要向仓库维护者获取；获取方式可通过本仓库Issues联系。未取得数据前，可以完整阅读报告并重绘已发布的聚合训练/评估图表，但不能仅凭代码重跑训练。新增数据、错误和类中心图需要相应数据/特征，见[可视化复现说明](report_visualizations.md)。

## 1. 下载代码

```bash
git clone https://github.com/tianzhaohaha/Meme_Foundation.git
cd Meme_Foundation
```

以下命令均在仓库根目录执行。无需拥有作者机器上的目录、缓存、checkpoint或W&B账号。

## 2. 创建环境

报告使用Linux x86_64、Python 3.12.0、PyTorch 2.8.0 + CUDA 12.8、torchvision 0.23.0和NVIDIA H200 NVL。对应版本依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-cu128.txt
python training/check_environment.py
```

历史PyTorch版本仅适合加载可信权重；已知依赖风险及较新环境选项见[主README](../README.md#创建环境)。依赖锁定文件记录当前实验环境的直接和传递依赖版本，不包含CUDA驱动、操作系统或GPU固件。不同硬件、框架版本和CUDA非确定性可能改变数值和耗时，不能保证逐位一致。

如果只重绘参考报告，不需要上述GPU环境：

```bash
python -m pip install -r requirements-analysis.txt
python tools/plot_results.py
```

## 3. 获取并核验同一份数据

将向维护者取得的原始包放到 `data/subset_50000.zip`。不要用自行重抽样的TumEmo子集替代本次50k包。参考包大小为4,295,027,951字节，完整校验值见 [dataset_signature.json](../data/dataset_signature.json)。

```bash
bash training/run_v1_direct.sh --prepare-only
python tools/verify_data.py --data data/subset_50000.zip
```

准备后为源训练20k、目标训练5k、验证2k和测试2k。校验同时核对记录顺序、标签、原图片哈希、清洗后文本指纹及已提交的验证/测试清单。不同机器的绝对缓存路径不参与该指纹。若校验失败，应先解决数据差异，不应将不同划分的结果宣称为本次复现。

首次训练需要联网取得ImageNet ViT和固定revision的RoBERTa模型/分词器。已有模型缓存可设置 `HF_HOME`、`TORCH_HOME`；只有缓存完整后才设置 `HF_HUB_OFFLINE=1`。

## 4. 串行运行六组实验

无W&B账号也可以运行：

```bash
WANDB_MODE=offline bash training/run_all_nohup.sh --suite all
bash training/run_all_nohup.sh status --suite all
tail -f training/background_all/queue.log
```

顺序为V1→V2→V3→V4→V5→V6。V2会先训练源20k二分类模型，V3/V5/V6自动复用同一个源checkpoint；不需要下载作者的实验checkpoint。输出及日志均位于本仓库下被Git排除的目录。

要与原报告保持相同在线日志模式，先执行 `.venv/bin/wandb login`，然后使用 `WANDB_MODE=online` 启动同一命令。W&B账号和项目应使用自己的配置，代码不包含作者凭据。日志模式会影响部分时间开销；同一组实验须保持一致。

中断和恢复：

```bash
bash training/run_all_nohup.sh stop --suite all
WANDB_MODE=offline bash training/run_all_nohup.sh --suite all -- --resume
```

停止会等待当前epoch保存；恢复应保持原数据、目录、参数及日志模式。SSH断开不影响nohup队列，服务器重启后需要手动恢复。不要同时启动其他实验争用同一GPU。

## 5. 查看并重绘自己的结果

```bash
python tools/collect_results.py
python tools/plot_results.py \
  --input training/runs_comparison/analysis/seed42.json \
  --output training/runs_comparison/analysis/figures
```

`training/runs_comparison/comparison_seed_42.csv` 和 `analysis/comparison_seed_42.csv` 提供机器可读对比；每阶段还有history、efficiency、test_metrics与配置。`best.pt`由验证集选出，报告测试的是best而不是last。重跑生成的新checkpoint无需加入Git。

已发布的历史结果位于 `reports/`，不会被以上命令覆盖。阅读[六版技术报告](../reports/experiment_review.md)时，同时对比accuracy、macro-F1、类别召回率、第二阶段耗时及包含源预训练的总成本。报告只有seed42，无法据此保证小幅差异具有统计显著性。

## 6. 验证代码与环境

```bash
python -m unittest discover -s tests -v
# 数据准备和模型缓存完成后，可选真实GPU单步检查；不会进行完整训练
python training/smoke_comparison_gpu.py
```

测试涵盖六组数据路由、融合层重置、冻结模块、断点恢复、后台队列、数据指纹和W&B离线记录。W&B离线服务仍需要本机socket权限。

## 7. 复现Qwen零样本validation评估

沿用同一环境及已准备的数据缓存即可运行，不需要先训练六版模型：

```bash
bash training/run_vlm_validation.sh
python tools/collect_vlm_results.py
python tools/plot_vlm_results.py \
  --input training/runs_vlm/analysis/vlm_validation.json \
  --output training/runs_vlm/analysis/figures
```

这会依次评估Qwen3-VL-4B和8B的固定Instruct版本，只使用validation。首次下载约27 GB；环境、输入限制、SSH后台启动与断点恢复见[VLM协议](vlm_validation.md)。只重绘已发布的VLM图表时，安装 `requirements-analysis.txt` 后执行 `python tools/plot_vlm_results.py` 即可，不需要模型或数据。
