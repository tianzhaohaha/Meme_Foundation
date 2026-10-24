# 实验报告与可视化

本目录集中保存已完成的分析报告、图表和支撑数据。**只阅读结果时，从本页开始，无需安装Python、下载数据或checkpoint，也无需登录W&B。** 在GitHub打开 `reports/` 即可直接阅读这个目录首页。

## 报告导航

| 想了解什么 | 打开哪份报告 | 评估范围 |
|---|---|---|
| 数据如何划分，各标签占多少，原始图文是什么样 | [Dataset-side数据集报告](dataset_review.md) | Source 20k、Target 5k、Validation 2k、Test 2k；固定样例 |
| 输入文本多长、多少为空或被截断，原图尺寸有什么差异 | [主报告开头：输入质量](experiment_review.md#输入质量与预处理影响) | 全部29,000条记录；另按划分和类别统计 |
| V1–V6哪个方案更准确、训练更快，第二阶段开销有什么区别 | [训练实验与结果分析](experiment_review.md) | 固定validation选模；报告独立test结果和训练成本 |
| Qwen3-VL 4B/8B直接做七分类的表现 | [Qwen零样本评估章节](experiment_review.md#qwen零样本七分类同一2000条validation) | 同一2,000条validation；不与test数值混排 |
| 五个模型分别在哪些样本上犯错，是否互补 | [主报告末尾：模型错误对照](experiment_review.md#模型错误对照同一validation上的共同错误与分歧) | V4/V5/V6、Qwen 4B/8B，同一2,000条validation |
| Binary、V4、V5、V6学到的融合特征有什么可见差异 | [特征可视化报告](embedding_visualization.md) | 同一2,000条validation；PCA/t-SNE与七类中心余弦 |
| 报告中的模型对应哪个checkpoint、哪一轮 | [Checkpoint文件索引](checkpoints.md) | 历史文件元数据，目录不包含权重 |

建议先看数据集报告，再读主实验报告，最后看特征可视化。PCA/t-SNE与类中心相似度用于描述特征分布，不能仅凭图形给模型排名或证明负迁移。

## 图表直达

- 数据：[标签分布](figures/dataset/label_distribution.png) · [原始样本](figures/dataset/sample_gallery.png) · [预处理对照](figures/dataset/preprocessing_examples.png) · [输入质量](figures/dataset/input_quality.png) · [分标签质量](figures/dataset/input_quality_by_label.png)。
- 监督实验：[准确率、成本与收敛](figures/comparison.png) · [PDF](figures/comparison.pdf) · [测试混淆矩阵](figures/confusion_matrices.png)。
- Qwen评估：[Validation质量与推理开销](figures/vlm_validation.png) · [PDF](figures/vlm_validation.pdf) · [混淆矩阵](figures/vlm_confusion_matrices.png)。
- 错误对照：[共同错误与分歧](figures/errors/error_overlap.png) · [图文案例及五模型预测](figures/errors/error_cases.png)。
- 融合特征：[PCA七类图](figures/embedding/fusion_pca_seven.png) · [t-SNE七类图](figures/embedding/fusion_tsne_seven.png) · [PCA极性图](figures/embedding/fusion_pca_binary.png) · [t-SNE极性图](figures/embedding/fusion_tsne_binary.png)。对应PDF、坐标CSV和投影设置在[embedding图表目录](figures/embedding/)。
- 类中心：[四模型7×7相似度热力图](figures/embedding/fusion_centroid_similarity.png) · [PDF](figures/embedding/fusion_centroid_similarity.pdf)。

## 支撑数据

| 内容 | 文件 |
|---|---|
| 六种训练策略的指标及逐轮记录 | [对比CSV](comparison.csv)、[聚合JSON](data/seed42.json) |
| 数据划分、类别数量和重复审计 | [数据审计JSON](data/dataset_audit.json) |
| 输入质量与样本选择 | [统计JSON](data/dataset_visualization.json)、[29k数值CSV](data/input_quality.csv)、[样本文本节选](data/sample_captions.md) |
| Qwen结果、标签预测与云端核验 | [对比CSV](data/vlm_validation.csv)、[聚合JSON](data/vlm_validation.json)、[预测标签CSV](data/vlm_predictions.csv)、[核验记录](data/vlm_wandb_verification.json) |
| 五模型错误对照 | [统计及案例选择](data/model_error_analysis.json)、[预测CSV](data/model_error_predictions.csv) |
| 特征提取和投影的可追溯记录 | [提取核验JSON](data/embedding_visualization.json)、[投影设置](figures/embedding/projection_settings.json) |
| 七个类中心的相似度 | [矩阵CSV](data/fusion_centroid_similarity.csv)、[定义及汇总JSON](data/fusion_centroid_similarity.json) |
| Checkpoint对应关系 | [CSV](checkpoints.csv)、[JSON](data/checkpoints.json) |

```text
reports/
  README.md                    # 报告总入口
  dataset_review.md            # 标签分布 + 原始样本 + 预处理对照
  experiment_review.md         # 输入质量 + 六版训练/Qwen评估 + 错误对照
  embedding_visualization.md   # Binary/V4/V5/V6的PCA、t-SNE与类中心
  checkpoints.md               # 权重文件与实验结果的对应关系
  comparison.csv               # 六版训练结果表
  checkpoints.csv              # Checkpoint元数据表
  figures/                     # PNG、PDF及embedding二维坐标
  data/                        # 聚合指标、审计和核验记录
```

图表和汇总数据可直接查看。完整训练的环境、数据获取和运行方法见[复现指南](../docs/reproduction.md)，具体配置见[实验协议](../docs/experiment_protocol.md)，新增图表生成方式见[可视化复现说明](../docs/report_visualizations.md)。本目录包含报告所用的少量样本缩略图及文本节选，不包含完整原始数据、checkpoint或高维特征矩阵。
