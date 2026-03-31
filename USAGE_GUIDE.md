# ToothSeg 使用指南 / Usage Guide

## 问题：是否需要迁移学习？/ Question: Do I Need Transfer Learning?

### 简短回答 / Short Answer

**不需要！** 如果你想使用ToothSeg模型对CBCT牙齿图像进行分割，**通常不需要迁移学习**。我们提供了预训练的模型权重，可以直接使用。

**No!** If you want to use the ToothSeg model for tooth segmentation in CBCT scans, **transfer learning is usually NOT required**. We provide pre-trained model weights that can be used directly.

---

## 详细说明 / Detailed Explanation

### 何时使用预训练模型（无需迁移学习）/ When to Use Pre-trained Models (No Transfer Learning)

✅ **推荐使用预训练模型的情况 / Use pre-trained models when:**

1. **数据类型相似** / Similar data type
   - 您的数据是CBCT牙齿扫描 / Your data is CBCT tooth scans
   - 成人或青少年牙齿 / Adult or adolescent dentition
   - 标准扫描协议 / Standard scanning protocols

2. **快速部署** / Quick deployment
   - 需要快速获得分割结果 / Need segmentation results quickly
   - 没有大量标注数据 / Don't have large annotated datasets
   - 评估模型是否适合您的应用场景 / Evaluating if the model fits your use case

3. **资源有限** / Limited resources
   - 计算资源有限 / Limited computational resources
   - 时间紧迫 / Time constraints
   - 没有深度学习训练经验 / No deep learning training experience

### 何时考虑迁移学习 / When to Consider Transfer Learning

⚠️ **可能需要迁移学习的情况 / Transfer learning may be beneficial when:**

1. **数据分布差异大** / Significant data distribution differences
   - 不同的扫描设备或参数 / Different scanning devices or parameters
   - 特殊人群（如儿童、老年人）/ Special populations (children, elderly)
   - 病理性牙齿结构 / Pathological tooth structures
   - 不同的图像质量或噪声水平 / Different image quality or noise levels

2. **性能需求** / Performance requirements
   - 预训练模型性能不满足要求 / Pre-trained model performance is insufficient
   - 需要针对特定数据集优化 / Need optimization for specific dataset
   - 有大量标注数据可用 / Large annotated dataset available

3. **特殊应用** / Special applications
   - 与标准CBCT扫描显著不同的成像模态 / Significantly different imaging modality from standard CBCT
   - 需要分割额外的结构 / Need to segment additional structures
   - 特定的临床应用需求 / Specific clinical application requirements

---

## 快速开始：使用预训练模型 / Quick Start: Using Pre-trained Models

### 第一步：下载预训练模型 / Step 1: Download Pre-trained Models

从Zenodo下载模型检查点：
Download model checkpoints from Zenodo:

```bash
# Download from: https://zenodo.org/records/14893540
# 下载两个文件夹 / Download two folders:
# - Dataset121_ToothFairy2_Teeth (语义分支 / Semantic branch)
# - Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px (实例分支 / Instance branch)
```

将下载的文件夹放入您的 `nnUNet_results` 目录：
Place the downloaded folders into your `nnUNet_results` directory:

```bash
mv Dataset121_ToothFairy2_Teeth $nnUNet_results/
mv Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px $nnUNet_results/
```

### 第二步：准备您的数据 / Step 2: Prepare Your Data

将您的测试图像放在 `imagesTs` 文件夹中：
Place your test images in an `imagesTs` folder:

```
your_data_folder/
└── imagesTs/
    ├── case_001_0000.nii.gz
    ├── case_002_0000.nii.gz
    └── ...
```

**重要提示 / Important Notes:**
- 图像格式必须是 NIfTI (.nii.gz) / Images must be in NIfTI format (.nii.gz)
- 文件名必须以 `_0000.nii.gz` 结尾 / Filenames must end with `_0000.nii.gz`
- 这是nnU-Net的标准格式要求 / This is the standard nnU-Net format requirement

### 第三步：运行推理 / Step 3: Run Inference

#### 方法一：使用简化的Python脚本（推荐）/ Method 1: Use Simplified Python Script (Recommended)

```bash
python toothseg/inference/simple_inference.py \
    --input_dir /path/to/your_data_folder \
    --output_dir /path/to/output \
    --gpu_id 0
```

这个脚本会自动完成所有步骤：
This script automatically handles all steps:
- 数据预处理（调整分辨率）/ Data preprocessing (resize)
- 双分支预测（语义+实例）/ Dual-branch prediction (semantic + instance)
- 后处理（格式转换、标签分配）/ Post-processing (format conversion, label assignment)

#### 方法二：使用原始bash脚本 / Method 2: Use Original Bash Script

```bash
# 编辑脚本中的路径 / Edit paths in the script
vim scripts/inference_generel.sh

# 设置 / Set:
# input_dir="/path/to/your_data_folder"
# output_dir="/path/to/output"

# 运行脚本 / Run script
bash scripts/inference_generel.sh
```

### 第四步：查看结果 / Step 4: View Results

分割结果将保存在：
Segmentation results will be saved in:

```
output_dir/
└── final_prediction/
    ├── case_001.nii.gz  # 每个牙齿都有唯一的实例ID和FDI标签
    ├── case_002.nii.gz  # Each tooth has unique instance ID and FDI label
    └── ...
```

每个体素的值代表：
Each voxel value represents:
- `0`: 背景 / Background
- `11-18, 21-28, 31-38, 41-48`: FDI牙位编号系统 / FDI tooth numbering system

---

## 高级用法：迁移学习 / Advanced: Transfer Learning

如果您决定需要迁移学习（即在您自己的数据上微调模型），请按照以下步骤：
If you decide transfer learning is needed (fine-tuning on your own data), follow these steps:

### 1. 准备标注数据 / Prepare Annotated Data

您需要标注的训练数据，格式遵循nnU-Net要求：
You need annotated training data following nnU-Net format:

```
DatasetXXX_YourDataset/
├── imagesTr/
│   ├── case_001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case_001.nii.gz  # 标注使用FDI编号 / Labels using FDI numbering
│   └── ...
└── dataset.json
```

### 2. 数据预处理 / Data Preprocessing

参考 [README.md](README.md) 的数据准备部分：
Refer to the Dataset Preparation section in [README.md](README.md):

- 创建参考数据集（原始分辨率）/ Create reference dataset (original spacing)
- 创建语义分支数据集（0.3x0.3x0.3）/ Create semantic branch dataset (0.3x0.3x0.3)
- 创建实例分支数据集（0.2x0.2x0.2，边界-核心格式）/ Create instance branch dataset (0.2x0.2x0.2, border-core format)

### 3. 使用预训练权重进行微调 / Fine-tune with Pre-trained Weights

```bash
# 语义分支微调 / Fine-tune semantic branch
nnUNetv2_train YOUR_DATASET_ID 3d_fullres_resample_torch_256_bs8 all \
    -tr nnUNetTrainer_onlyMirror01_DASegOrd0 \
    -pretrained_weights $nnUNet_results/Dataset121_ToothFairy2_Teeth/.../checkpoint_final.pth \
    -num_gpus 4

# 实例分支微调 / Fine-tune instance branch
nnUNetv2_train YOUR_DATASET_ID 3d_fullres_resample_torch_192_bs8 all \
    -tr nnUNetTrainer \
    -pretrained_weights $nnUNet_results/Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px/.../checkpoint_final.pth \
    -num_gpus 4
```

### 4. 评估微调后的模型 / Evaluate Fine-tuned Model

使用验证集评估模型性能，比较微调前后的结果：
Evaluate model performance on validation set, compare before and after fine-tuning:

```bash
# 运行推理 / Run inference
bash scripts/inference_generel.sh

# 评估 / Evaluate
python toothseg/evaluation/evaluate_instances_with_tooth_label.py \
    -gt /path/to/ground_truth \
    -pred /path/to/predictions
```

---

## 性能预期 / Expected Performance

### 预训练模型（ToothFairy2数据集）/ Pre-trained Model (ToothFairy2 Dataset)

基于ToothFairy2挑战数据集的性能：
Performance on ToothFairy2 Challenge Dataset:

- **Dice相似系数 / Dice Similarity Coefficient**: ~0.92-0.94
- **FDI准确率 / FDI Accuracy**: ~0.95-0.97
- **实例检测率 / Instance Detection Rate**: >0.98

**注意 / Note**: 实际性能取决于您的数据与训练数据的相似度
Actual performance depends on similarity between your data and training data

### 何时性能可能下降 / When Performance May Degrade

- 图像质量显著较差 / Significantly worse image quality
- 不同的扫描协议或设备 / Different scanning protocols or devices
- 特殊人群或病理情况 / Special populations or pathological cases
- 严重的金属伪影 / Severe metal artifacts

**解决方案 / Solutions**:
1. 首先在少量样本上测试预训练模型 / First test pre-trained model on small sample
2. 如果性能不佳，考虑收集标注数据进行微调 / If performance is poor, consider collecting annotated data for fine-tuning
3. 可以从少量数据开始（50-100个病例）/ Can start with small dataset (50-100 cases)

---

## 常见问题 / FAQ

### Q1: 我的数据格式是DICOM，怎么办？
**A**: 需要将DICOM转换为NIfTI格式。可以使用以下工具：
- SimpleITK: `sitk.WriteImage(sitk.ReadImage(dicom_series), output.nii.gz)`
- dcm2niix: `dcm2niix -o output_dir input_dicom_dir`

### Q1: My data is in DICOM format, what should I do?
**A**: Convert DICOM to NIfTI format using:
- SimpleITK: `sitk.WriteImage(sitk.ReadImage(dicom_series), output.nii.gz)`
- dcm2niix: `dcm2niix -o output_dir input_dicom_dir`

---

### Q2: 推理需要多长时间？
**A**: 在单个GPU上，每个病例大约需要2-5分钟，具体取决于图像大小和GPU性能。

### Q2: How long does inference take?
**A**: On a single GPU, approximately 2-5 minutes per case, depending on image size and GPU performance.

---

### Q3: 需要什么样的GPU？
**A**: 推荐使用至少11GB显存的GPU（如NVIDIA RTX 2080 Ti或更好）。可以通过减小批量大小在较小的GPU上运行。

### Q3: What GPU do I need?
**A**: Recommended GPU with at least 11GB VRAM (e.g., NVIDIA RTX 2080 Ti or better). Can run on smaller GPUs by reducing batch size.

---

### Q4: 可以在CPU上运行吗？
**A**: 技术上可以，但会非常慢（每个病例可能需要30-60分钟或更长）。强烈推荐使用GPU。

### Q4: Can I run on CPU?
**A**: Technically yes, but very slow (30-60+ minutes per case). GPU is strongly recommended.

---

### Q5: 如何知道是否需要迁移学习？
**A**:
1. 先在少量样本（5-10个病例）上测试预训练模型
2. 如果分割结果令人满意，无需迁移学习
3. 如果结果不佳，且您有标注数据，考虑迁移学习
4. 即使只有50-100个标注样本也可能带来显著改进

### Q5: How do I know if I need transfer learning?
**A**:
1. First test pre-trained model on small sample (5-10 cases)
2. If segmentation results are satisfactory, no transfer learning needed
3. If results are poor and you have annotated data, consider transfer learning
4. Even 50-100 annotated samples can bring significant improvements

---

### Q6: 模型可以处理缺失的牙齿吗？
**A**: 是的！模型经过训练可以处理各种牙齿情况，包括缺失牙齿、种植体和修复体。

### Q6: Can the model handle missing teeth?
**A**: Yes! The model is trained to handle various dental conditions including missing teeth, implants, and restorations.

---

### Q7: 输出的FDI编号是什么？
**A**: FDI（国际牙科联合会）编号系统：
- 11-18: 右上象限 / Upper right quadrant
- 21-28: 左上象限 / Upper left quadrant
- 31-38: 左下象限 / Lower left quadrant
- 41-48: 右下象限 / Lower right quadrant

### Q7: What is the FDI numbering in the output?
**A**: FDI (Fédération Dentaire Internationale) numbering system:
- 11-18: Upper right quadrant
- 21-28: Upper left quadrant
- 31-38: Lower left quadrant
- 41-48: Right lower quadrant

---

## 技术支持 / Technical Support

如果遇到问题，请：
If you encounter issues:

1. 检查图像格式是否正确（NIfTI格式，`_0000.nii.gz`后缀）
   Check image format is correct (NIfTI format, `_0000.nii.gz` suffix)

2. 确认已正确设置环境变量 `nnUNet_results`
   Confirm environment variable `nnUNet_results` is set correctly

3. 查看[主README](README.md)了解详细的安装和使用说明
   See [main README](README.md) for detailed installation and usage instructions

4. 在GitHub Issues中报告问题并提供错误信息
   Report issues on GitHub Issues with error messages

---

## 引用 / Citation

如果您在研究中使用ToothSeg，请引用：
If you use ToothSeg in your research, please cite:

```bibtex
@ARTICLE{toothseg,
  author={van Nistelrooij, Niels and Krämer, Lars and Kempers, Steven and Beyer, Michel and Bolelli, Federico and Xi, Tong and Bergé, Stefaan and Heiland, Max and Maier-Hein, Klaus H. and Vinayahalingam, Shankeeth and Isensee, Fabian},
  journal={IEEE Journal of Biomedical and Health Informatics},
  title={ToothSeg: Robust Tooth Instance Segmentation and Numbering in CBCT using Deep Learning and Self-Correction},
  year={2025},
  doi={10.1109/JBHI.2025.3650444}
}
```
