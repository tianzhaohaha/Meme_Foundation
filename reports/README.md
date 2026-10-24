# 实验报告与可视化

本目录集中保存已完成的分析报告、图表和支撑数据。**只阅读结果时，从本页开始，无需安装Python、下载数据或checkpoint，也无需登录W&B。** 在GitHub打开 `reports/` 即可直接阅读这个目录首页。

## 报告导航

| 想了解什么 | 打开哪份报告 | 评估范围 |
|---|---|---|
| V1–V6哪个方案更准确、训练更快，第二阶段开销有什么区别 | [训练实验与结果分析](experiment_review.md) | 固定validation选模；报告独立test结果和训练成本 |
| Qwen3-VL 4B/8B直接做七分类的表现 | [Qwen零样本评估章节](experiment_review.md#qwen零样本七分类同一2000条validation) | 同一2,000条validation；不与test数值混排 |
| Binary、V4、V5、V6学到的融合特征有什么可见差异 | [PCA与t-SNE可视化报告](embedding_visualization.md) | 同一2,000条validation；七类与极性着色 |
| 报告中的模型对应哪个checkpoint、哪一轮 | [Checkpoint文件索引](checkpoints.md) | 历史文件元数据，目录不包含权重 |

建议先读训练实验报告，再看零样本评估和特征可视化。PCA/t-SNE用于描述特征分布，不能仅凭二维簇形状给模型排名或证明负迁移。

## 图表直达

- 监督实验：[准确率、成本与收敛](figures/comparison.png) · [PDF](figures/comparison.pdf) · [测试混淆矩阵](figures/confusion_matrices.png)。
- Qwen评估：[Validation质量与推理开销](figures/vlm_validation.png) · [PDF](figures/vlm_validation.pdf) · [混淆矩阵](figures/vlm_confusion_matrices.png)。
- 融合特征：[PCA七类图](figures/embedding/fusion_pca_seven.png) · [t-SNE七类图](figures/embedding/fusion_tsne_seven.png) · [PCA极性图](figures/embedding/fusion_pca_binary.png) · [t-SNE极性图](figures/embedding/fusion_tsne_binary.png)。对应PDF、坐标CSV和投影设置在[embedding图表目录](figures/embedding/)。

## 支撑数据

| 内容 | 文件 |
|---|---|
| 六种训练策略的指标及逐轮记录 | [对比CSV](comparison.csv)、[聚合JSON](data/seed42.json) |
| 数据划分、类别数量和重复审计 | [数据审计JSON](data/dataset_audit.json) |
| Qwen结果、标签预测与云端核验 | [对比CSV](data/vlm_validation.csv)、[聚合JSON](data/vlm_validation.json)、[预测标签CSV](data/vlm_predictions.csv)、[核验记录](data/vlm_wandb_verification.json) |
| 特征提取和投影的可追溯记录 | [提取核验JSON](data/embedding_visualization.json)、[投影设置](figures/embedding/projection_settings.json) |
| Checkpoint对应关系 | [CSV](checkpoints.csv)、[JSON](data/checkpoints.json) |

```text
reports/
  README.md                    # 报告总入口
  experiment_review.md         # 六版训练 + Qwen零样本评估
  embedding_visualization.md   # Binary/V4/V5/V6的PCA与t-SNE
  checkpoints.md               # 权重文件与实验结果的对应关系
  comparison.csv               # 六版训练结果表
  checkpoints.csv              # Checkpoint元数据表
  figures/                     # PNG、PDF及embedding二维坐标
  data/                        # 聚合指标、审计和核验记录
```

图表和汇总数据可直接查看。完整训练的环境、数据获取和运行方法见[复现指南](../docs/reproduction.md)，具体配置见[实验协议](../docs/experiment_protocol.md)。原始图片/文本、checkpoint和高维特征矩阵不在本目录中。
