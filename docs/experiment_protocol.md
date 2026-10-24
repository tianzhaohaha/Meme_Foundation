# 实验协议

## 模型和三个训练方案

默认模型为：ImageNet预训练ViT-B/16（768维）+ `FacebookAI/roberta-base`（attention-mask均值池化，768维）→ 拼接1536维 → Linear256 / LayerNorm / ReLU → Dropout0.5 / Linear分类头。

| 方案 | 第一阶段 | 第二阶段 |
|---|---|---|
| Direct | 20,000源样本+5,000目标样本，直接七分类，全参数训练 | 无 |
| Full | 20,000源样本二分类，全参数训练 | 替换新七分类头，在5,000样本上整体微调 |
| Frozen | 与Full共享同一个二分类checkpoint | 替换新七分类头，冻结两个编码器及融合层，仅训练新头 |

第二阶段分别创建新的优化器和调度器；保留源checkpoint中的编码器和融合层权重，丢弃二分类头。Full/Frozen的七分类头初始化、seed和样本顺序相同。冻结主干保持eval模式、关闭主干梯度，分类头仍为训练模式。特征没有离线缓存，每个batch仍需执行ViT和RoBERTa前向。

最终三版都包含约2.10亿参数。Frozen只有1,799个参数参与更新，不等于推理模型被压缩成1,799个参数。可选 `--architecture vit`用于另一组纯图像实验；不得与本报告的图文结果混合。

## 数据

七类固定顺序为Angry、Bored、Calm、Fear、Happy、Love、Sad。二分类正类为Calm/Happy/Love，负类为其他四类。

| 类别 | 源20k | 目标5k | 验证2k | 测试2k |
|---|---:|---:|---:|---:|
| Angry | 1505 | 376 | 116 | 116 |
| Bored | 3298 | 825 | 356 | 357 |
| Calm | 1828 | 457 | 187 | 187 |
| Fear | 2050 | 513 | 215 | 215 |
| Happy | 5160 | 1290 | 557 | 557 |
| Love | 3533 | 883 | 346 | 346 |
| Sad | 2626 | 656 | 223 | 222 |
| 合计 | 20000 | 5000 | 2000 | 2000 |

源/目标清单ID不重叠。验证/测试从剩余总池中按七类分层抽取，split seed42，排除与训练数据重复的图片字节哈希和清洗、大小写折叠、空白规范化后的非空文本；验证与测试之间也不重复。原始训练清单内部重复保留，其中源与目标共有227个图片哈希。空文本仍作为缺失文本样本参与训练/评估；没有声称已完成语义近重复或用户分组审计。

图片等比例填充至224×224并按ImageNet均值/标准差归一化；文本去除URL及情绪hashtag，最多96 tokens。标签和polarity字段不作为模型输入。测试集不参与早停和模型选择；二分类阶段不评估测试集。

## 默认训练配置

| 参数 | 值 |
|---|---|
| 优化器 | Adam，非AdamW |
| 编码器LR | 5e-6 |
| 融合层/分类头LR | 1e-4 |
| weight decay | 0.01 |
| warmup / schedule | 总计划步数的5% / cosine |
| batch / precision | 16 / FP32，未启用AMP |
| 最大epoch / patience | 每阶段20 / 4 |
| 损失 | 交叉熵，无类别加权，无label smoothing |
| gradient clipping | max norm 1.0 |
| DataLoader workers / torch CPU线程 | 0 / 4 |
| 训练seed / 划分seed | 42 / 42 |
| 二分类选模指标 | 验证macro-F1 |
| 七分类选模指标 | 验证accuracy |

`best.pt`按选模指标严格提升时更新；同分保留更早轮次。`best_accuracy.pt`始终按accuracy选择，`last.pt`用于恢复。早停检查每epoch一次。本次实际为源二分类7轮、Direct6轮、Full第二阶段6轮、Frozen第二阶段20轮。

## 效率与W&B

每阶段一个run；共享二分类标记 `variant=shared_binary, stage=binary`，最终三版为 `direct/direct`、`full/stage2`、`frozen/stage2`。每25步及epoch末记录指标，所有最终run摘要使用共同 `comparison/*` 字段。

- `train_seconds`计训练循环，包括数据读取、GPU计算、优化和定期日志；使用单调时钟，并在GPU阶段边界同步。
- `validation_seconds`、`checkpoint_seconds`分别记录验证及保存成本。
- `stage_wall_seconds`计设置、训练、验证、保存与最终测试；不包含外部数据预处理、停机等待、W&B最后上传等待和进程启动外部开销。
- `total_train_seconds`、`total_stage_wall_seconds`对Full/Frozen均计入二分类原始成本；`stage2_*`只比较微调阶段。
- 峰值显存覆盖epoch训练与验证；整个Frozen方案的峰值仍包含源预训练，不能只引用第二阶段低显存。
- 按验证accuracy随训练时间绘制收敛曲线；默认达标阈值80%，未达到时为空值。本次三个七分类模型均未达到。

比较目录校验共同数据、架构、超参数、框架版本、GPU型号、batch和日志模式。GPU上的其他任务仍会影响计时，应使用空闲GPU并串行运行。中断恢复累计已完成epoch；丢失的半轮耗时无法恢复，不能把累计值解释为包含所有失败尝试的账单成本。

## 可重复性与结论范围

本次是单seed、单设备结果。V1采用25,000条七分类监督，Full/Frozen只采用5,000条七分类监督加源二分类标签；整体对比是完整方案比较，Full与Frozen第二阶段才更接近只改变冻结策略的对照。

多个seed实验可以使用同一个输出根目录和固定划分，但每次队列要完成后再启动下一seed，或使用独立日志目录。示例：`bash training/run_all_nohup.sh --log-dir logs_seed123 -- --seed 123`。使用共同验证规则选模，对测试集只做最终汇报，避免以测试集反复调参。

聚合报告保留了本次原始数据指纹和source checkpoint哈希。预处理记录包含绝对图片路径，迁移目录会改变指纹；可通过已提交的验证/测试ID与图片哈希核对划分一致，但旧运行目录的断点恢复应在原路径进行。
