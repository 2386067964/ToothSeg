# ToothSeg 快速参考 / Quick Reference

## 核心问题回答 / Core Question Answered

### ❓ 如果我想要利用这个分割模型进行分割，是否需要迁移学习？

### ✅ 答案：大多数情况下**不需要**！/ Answer: Most cases **DON'T need it**!

---

## 三种使用场景 / Three Usage Scenarios

### 🚀 场景 1：直接使用（推荐，90%的用户）/ Scenario 1: Direct Use (Recommended, 90% of users)

**适用于 / Suitable for:**
- 标准CBCT牙齿扫描 / Standard CBCT tooth scans
- 快速评估或部署 / Quick evaluation or deployment
- 没有大量标注数据 / No large annotated dataset

**步骤 / Steps:**
```bash
# 1. 下载预训练模型 / Download pre-trained models
# 从 Zenodo: https://zenodo.org/records/14893540

# 2. 运行推理 / Run inference
python toothseg/inference/simple_inference.py \
    --input_dir /path/to/your_data \
    --output_dir /path/to/output \
    --gpu_id 0
```

**优点 / Advantages:**
- ✅ 无需训练 / No training needed
- ✅ 快速部署（几分钟内开始使用）/ Quick deployment (start using in minutes)
- ✅ 已在多个数据集上验证 / Validated on multiple datasets
- ✅ 性能通常很好 / Performance usually good

---

### 🔧 场景 2：先测试再决定 / Scenario 2: Test First, Then Decide

**适用于 / Suitable for:**
- 不确定模型是否适合您的数据 / Unsure if model suits your data
- 有一些标注数据可以评估 / Have some annotated data for evaluation

**步骤 / Steps:**
```bash
# 1. 先在少量数据上测试预训练模型 / First test on small sample
python toothseg/inference/simple_inference.py \
    --input_dir /path/to/test_cases \
    --output_dir /path/to/output

# 2. 评估结果 / Evaluate results
python toothseg/evaluation/evaluate_instances_with_tooth_label.py \
    -gt /path/to/ground_truth \
    -pred /path/to/output/final_prediction

# 3. 根据结果决定是否需要微调 / Decide based on results
# 如果 Dice > 0.90: 直接使用 / If Dice > 0.90: use directly
# 如果 Dice < 0.85: 考虑微调 / If Dice < 0.85: consider fine-tuning
# 如果 0.85 < Dice < 0.90: 取决于应用需求 / If 0.85 < Dice < 0.90: depends on requirements
```

---

### 🎯 场景 3：迁移学习/微调（高级用户）/ Scenario 3: Transfer Learning/Fine-tuning (Advanced)

**何时需要 / When needed:**
- ⚠️ 预训练模型性能不佳（Dice < 0.85）/ Pre-trained model performs poorly (Dice < 0.85)
- ⚠️ 特殊数据分布（不同设备、特殊人群）/ Special data distribution (different devices, special populations)
- ⚠️ 有大量标注数据可用（>50个病例）/ Large annotated dataset available (>50 cases)

**步骤 / Steps:**
```bash
# 1. 准备数据集 / Prepare dataset
# 参考 README.md 的 "Dataset Preparation" 部分

# 2. 使用预训练权重微调 / Fine-tune with pre-trained weights
nnUNetv2_train YOUR_DATASET_ID 3d_fullres_resample_torch_256_bs8 all \
    -tr nnUNetTrainer_onlyMirror01_DASegOrd0 \
    -pretrained_weights $nnUNet_results/Dataset121_ToothFairy2_Teeth/.../checkpoint_final.pth \
    -num_gpus 4

# 3. 评估改进 / Evaluate improvement
# 比较微调前后的性能
```

**注意 / Note:**
- 需要深度学习经验 / Requires deep learning experience
- 需要计算资源（GPU）/ Requires computational resources (GPUs)
- 训练时间：1-3天（取决于数据量）/ Training time: 1-3 days (depends on data size)

---

## 快速决策树 / Quick Decision Tree

```
开始 / Start
    │
    ├─ 有标注的测试数据？/ Have annotated test data?
    │   │
    │   ├─ 是 / Yes → 先测试预训练模型 / Test pre-trained model first
    │   │              │
    │   │              ├─ Dice > 0.90 → ✅ 直接使用 / Use directly
    │   │              ├─ 0.85 < Dice < 0.90 → 根据需求决定 / Decide based on needs
    │   │              └─ Dice < 0.85 → 考虑微调 / Consider fine-tuning
    │   │
    │   └─ 否 / No → ✅ 直接使用预训练模型 / Use pre-trained model directly
    │                 然后在实际使用中评估 / Evaluate during actual use
    │
    └─ 数据是否非常特殊？/ Is data very special?
        │
        ├─ 是 / Yes (不同设备/人群) → 可能需要微调 / May need fine-tuning
        │                              但先测试预训练模型 / But test pre-trained first
        │
        └─ 否 / No (标准CBCT) → ✅ 直接使用预训练模型 / Use pre-trained model directly
```

---

## 常见误解 / Common Misconceptions

### ❌ 误解 1 / Misconception 1
"深度学习模型总是需要用自己的数据训练"
"Deep learning models always need training on own data"

### ✅ 真相 / Truth
预训练模型在相似任务上已经学到了通用特征，可以直接使用
Pre-trained models learned general features on similar tasks, can be used directly

---

### ❌ 误解 2 / Misconception 2
"迁移学习总是能提高性能"
"Transfer learning always improves performance"

### ✅ 真相 / Truth
- 如果数据相似：预训练模型已经足够好 / If data similar: pre-trained model already good enough
- 如果数据不同：迁移学习可能有帮助 / If data different: transfer learning may help
- 需要充足的标注数据（建议>50个病例）/ Need sufficient annotated data (recommend >50 cases)

---

### ❌ 误解 3 / Misconception 3
"微调很简单，应该总是尝试"
"Fine-tuning is simple, should always try"

### ✅ 真相 / Truth
微调需要：
Fine-tuning requires:
- 大量标注数据 / Large annotated dataset
- 深度学习经验 / Deep learning experience
- 计算资源和时间 / Computational resources and time
- 超参数调整 / Hyperparameter tuning

先测试预训练模型！/ Test pre-trained model first!

---

## 性能参考 / Performance Reference

### ToothFairy2 数据集上的性能 / Performance on ToothFairy2 Dataset

| 指标 / Metric | 预训练模型 / Pre-trained Model |
|--------------|------------------------------|
| Dice系数 / Dice Coefficient | 0.92-0.94 |
| FDI准确率 / FDI Accuracy | 0.95-0.97 |
| 实例检测率 / Instance Detection | >0.98 |

### 性能可能下降的情况 / When Performance May Degrade

- 图像质量显著较差 / Significantly worse image quality
- 不同的扫描设备或协议 / Different scanning devices or protocols
- 严重的金属伪影 / Severe metal artifacts
- 特殊人群（如儿童牙列）/ Special populations (e.g., pediatric dentition)

**解决方案 / Solution**: 先测试，不满意再微调 / Test first, fine-tune only if unsatisfied

---

## 推荐工作流程 / Recommended Workflow

### 第一周 / Week 1: 测试预训练模型 / Test Pre-trained Model
```bash
# 准备5-10个测试病例 / Prepare 5-10 test cases
# 运行推理 / Run inference
python toothseg/inference/simple_inference.py --input_dir ... --output_dir ...

# 人工检查结果 / Manually inspect results
# 如果有标注数据，计算Dice系数 / If have annotations, compute Dice
```

### 决策点 / Decision Point
- ✅ 结果满意？→ 直接部署使用 / Results satisfactory? → Deploy and use
- ⚠️ 结果不理想？→ 继续下一步 / Results unsatisfactory? → Continue to next step

### 第二周起 / Week 2+: 考虑微调（如果需要）/ Consider Fine-tuning (if needed)
```bash
# 收集标注数据（建议>50个病例）/ Collect annotations (recommend >50 cases)
# 准备数据集 / Prepare dataset
# 开始微调 / Start fine-tuning
# 比较性能 / Compare performance
```

---

## 资源需求对比 / Resource Requirements Comparison

| 方式 / Approach | 数据需求 / Data | 时间 / Time | GPU需求 / GPU | 经验要求 / Experience |
|----------------|----------------|------------|-------------|-------------------|
| 直接使用预训练模型 / Direct Use | 无标注 / No labels | 分钟 / Minutes | 推理级 / Inference | 基础 / Basic |
| 微调（迁移学习）/ Fine-tuning | >50个标注病例 / >50 labeled | 1-3天 / 1-3 days | 训练级 / Training | 高级 / Advanced |
| 从头训练 / Train from Scratch | >200个标注病例 / >200 labeled | 1-2周 / 1-2 weeks | 训练级 / Training | 专家 / Expert |

---

## 联系与支持 / Contact & Support

- 📖 详细文档 / Detailed docs: [USAGE_GUIDE.md](USAGE_GUIDE.md)
- 📝 主README / Main README: [README.md](README.md)
- 🔬 论文 / Paper: [IEEE JBHI 2025](https://doi.org/10.1109/JBHI.2025.3650444)
- 💾 模型下载 / Model download: [Zenodo](https://zenodo.org/records/14893540)
- 🐛 问题报告 / Issue reporting: GitHub Issues

---

## 总结 / Summary

### 🎯 关键信息 / Key Takeaway

**90%的用户应该直接使用预训练模型，无需迁移学习！**
**90% of users should use pre-trained models directly, no transfer learning needed!**

只有在以下情况才考虑迁移学习：
Only consider transfer learning when:
1. 预训练模型性能明确不足 / Pre-trained model clearly underperforms
2. 有充足的标注数据（>50个病例）/ Have sufficient annotated data (>50 cases)
3. 有深度学习训练经验和资源 / Have DL training experience and resources

**建议顺序 / Recommended Order:**
测试预训练 → 评估性能 → 决定是否微调
Test pre-trained → Evaluate → Decide if fine-tuning needed

---

*最后更新 / Last updated: 2026-03-31*
