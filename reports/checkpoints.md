# Checkpoint 对应关系

以下21个文件均已读取实际checkpoint元数据核对。路径相对于**作者原项目的 `TumEmo_Binary_Training_Handoff/` 目录**；本页是历史结果的文件索引，GitHub仓库不提供这些权重。按复现指南重新训练后，自己的checkpoint默认写入 `training/runs_comparison/`。

## 六个最终七分类模型

| 用途 | 文件相对路径 | epoch | 验证accuracy | 该最佳模型测试accuracy |
|---|---|---:|---:|---:|
| V1推荐评估/推理 | `runs_comparison/direct/seed_42/best.pt` | 2 | 57.55% | **57.65%** |
| V1按accuracy选出的副本名称 | `runs_comparison/direct/seed_42/best_accuracy.pt` | 2 | 57.55% | 同上 |
| V1断点恢复 | `runs_comparison/direct/seed_42/last.pt` | 6 | 50.25% | 未单独测试 |
| V2推荐评估/推理 | `runs_comparison/full/seed_42/best.pt` | 2 | 54.65% | **54.30%** |
| V2按accuracy选出的副本名称 | `runs_comparison/full/seed_42/best_accuracy.pt` | 2 | 54.65% | 同上 |
| V2断点恢复 | `runs_comparison/full/seed_42/last.pt` | 6 | 53.50% | 未单独测试 |
| V3推荐评估/推理 | `runs_comparison/frozen/seed_42/best.pt` | 18 | 41.65% | **42.10%** |
| V3按accuracy选出的副本名称 | `runs_comparison/frozen/seed_42/best_accuracy.pt` | 18 | 41.65% | 同上 |
| V3断点恢复 | `runs_comparison/frozen/seed_42/last.pt` | 20 | 41.65% | 未单独测试 |
| V4推荐评估/推理 | `runs_comparison/direct_5k/seed_42/best.pt` | 8 | 56.50% | 56.90% |
| V4按accuracy选出的副本名称 | `runs_comparison/direct_5k/seed_42/best_accuracy.pt` | 8 | 56.50% | 56.90% |
| V4断点恢复 | `runs_comparison/direct_5k/seed_42/last.pt` | 12 | 55.00% | 未单独测试 |
| V5推荐评估/推理 | `runs_comparison/reset_fusion/seed_42/best.pt` | 4 | 54.80% | 54.85% |
| V5按accuracy选择 | `runs_comparison/reset_fusion/seed_42/best_accuracy.pt` | 4 | 54.80% | 54.85% |
| V5断点恢复 | `runs_comparison/reset_fusion/seed_42/last.pt` | 8 | 53.05% | 未单独测试 |
| V6推荐评估/推理 | `runs_comparison/tune_fusion/seed_42/best.pt` | 4 | 56.30% | 54.75% |
| V6按accuracy选择 | `runs_comparison/tune_fusion/seed_42/best_accuracy.pt` | 4 | 56.30% | 54.75% |
| V6断点恢复 | `runs_comparison/tune_fusion/seed_42/last.pt` | 8 | 56.25% | 未单独测试 |

V3第20轮验证accuracy与第18轮持平，但模型参数和macro-F1不同；最终42.10%的测试结果对应第18轮 `best.pt`，不能无验证地归到 `last.pt`。

## V2/V3/V5/V6共享的二分类源模型

| 用途 | 文件相对路径 | epoch | 验证accuracy / macro-F1 |
|---|---|---:|---|
| 四个第二阶段实际加载的源模型 | `runs_comparison/shared_binary/e1555ba804919d21/seed_42/best.pt` | 3 | 81.75% / 0.816956 |
| 按accuracy选出的源模型 | `runs_comparison/shared_binary/e1555ba804919d21/seed_42/best_accuracy.pt` | 3 | 81.75% / 0.816956 |
| 二分类阶段断点恢复 | `runs_comparison/shared_binary/e1555ba804919d21/seed_42/last.pt` | 7 | 79.55% / 0.793242 |

共享源模型 `best.pt` 的SHA-256为：

```text
a098806de821c9d0fde3d90399fa5ef4de06fff4b9c774de6495a97b0d18ee53
```

V2、V3、V5、V6的配置记录了相同来源哈希，源文件已重新计算哈希核对。二分类模型只有2个输出，不能直接当作七分类最终模型使用。

## 文件内容与恢复

本次七个阶段中，原训练目录内每一对 `best.pt` / `best_accuracy.pt` 都指向同一inode，是同一份内容的硬链接。`last.pt`为不同epoch的另一份文件。以后若按macro-F1与accuracy选到不同轮次，两个best文件可以不同。

- V1/V2/V4/V5完整checkpoint约2.35 GiB，包含全模型及Adam状态。
- V6完整checkpoint约0.79 GiB，包含完整模型及融合层/分类头的Adam状态。
- V3完整checkpoint约0.78 GiB，主干仍完整存在，文件较小主要因为只保存分类头的Adam状态；它不是只有1,799个参数的小型推理模型。
- `best.pt`用于报告中的测试结果与后续评估；`last.pt`用于原目录下继续训练。
- `runs_binary/seed_42/*.pt`是更早的旧二分类实验，划分不同；`runs_smoke/seed_42/*.pt`是冒烟测试，均不是本次六版最终结果。

加载七分类权重示例（从仓库根目录执行，`checkpoint_path`指向自行训练得到的可信本地文件）：

```python
import sys
sys.path.insert(0, 'training')
from model import EmotionModel
from training_engine import load_checkpoint

checkpoint_path = '/absolute/path/to/runs_comparison/direct/seed_42/best.pt'
saved = load_checkpoint(checkpoint_path)
model = EmotionModel(pretrained=False,
                     num_classes=saved['config']['num_classes'],
                     architecture=saved['config']['architecture'],
                     dropout=saved['config']['dropout'])
model.load_state_dict(saved['model_state_dict'])
model.eval()
```

推理仍需原图像预处理、相同RoBERTa tokenizer和attention mask。原实验断点应在原项目路径恢复；预处理manifest含绝对路径，不要把旧checkpoint移入新目录后期待无条件续训。

机器可读清单见 [checkpoints.csv](checkpoints.csv) 和 [JSON](data/checkpoints.json)，包含文件大小、选择指标、类别数、epoch、来源哈希及与best是否同一文件。
