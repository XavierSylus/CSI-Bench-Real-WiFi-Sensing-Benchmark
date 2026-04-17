# CSI-Bench Motion Source Recognition (MSR) 四分类复现阶段总结

## 1. 任务定义核查结论
在复现过程中，首先发现公开默认配置下的 `MotionSourceRecognition` 实际读取的是：
- `metadata/label_mapping.json`

该文件将原始四类：
- Fan
- Human
- IRobot
- Pet

合并为二类：
- Nonhuman
- Human

因此，仓库默认训练入口实际执行的是 **Human vs Nonhuman 二分类任务**，而不是论文主表对应的 **四分类 Motion Source Recognition**。

进一步核查发现，同目录下还存在：
- `metadata/label_mapping_fourclass.json`

其中保留了论文定义一致的四类映射：
- Fan -> 0
- Human -> 1
- IRobot -> 2
- Pet -> 3

因此，后续实验将 `label_mapping.json` 切换为四分类映射后，重新进行正式 baseline 训练。

---

## 2. 四分类 MSR baseline 复现结果

### 2.1 Transformer
- Test Accuracy: 0.9508
- Test F1-score: 0.9510
- Easy Accuracy/F1: 0.9491 / 0.9570
- Medium Accuracy/F1: 0.9559 / 0.9563
- Hard Accuracy/F1: 0.9356 / 0.9365

### 2.2 ResNet18
- Test Accuracy: 0.9963
- Test F1-score: 0.9963
- Easy Accuracy/F1: 0.9954 / 0.9954
- Medium Accuracy/F1: 0.9971 / 0.9971
- Hard Accuracy/F1: 0.9940 / 0.9940

---

## 3. 与论文结果对照（Motion Source Recognition, 4-class）

### 3.1 论文报告结果
论文中，MSR 是四分类任务（human / pet / robot / fan）。

#### Transformer
- Main Test Acc/F1: 98.61 / 98.61
- Easy Acc/F1: 98.73 / 98.80
- Medium Acc/F1: 98.63 / 98.63
- Hard Acc/F1: 98.08 / 98.08

#### ResNet18
- Main Test Acc/F1: 99.56 / 99.56
- Easy Acc/F1: 99.86 / 99.86
- Medium Acc/F1: 99.73 / 99.73
- Hard Acc/F1: 99.48 / 99.48

### 3.2 当前复现判断
- Transformer: 已跑通四分类正式 baseline，但与论文结果仍有明显 gap
- ResNet18: 与论文结果非常接近，可视为高保真单任务 baseline 复现

---

## 4. 当前阶段判断
当前已完成：
- 单任务监督学习链路跑通
- MSR 任务定义核查
- 四分类 mapping 切换验证
- Transformer 四分类 baseline
- ResNet18 四分类 baseline

当前阶段应定义为：

**CSI-Bench 单任务 benchmark 复现进入“基线级复现初步达成”阶段。**

其中：
- MSR + ResNet18：高保真复现
- MSR + Transformer：有效复现，但仍需进一步调参对齐

---

## 5. 已知问题
当前结果保存逻辑仍存在一个非关键 bug：
- `classification_report_unknown.csv` 会重复出现
- split 名没有正确写入报告文件名

该问题目前不影响主结果判断，但后续整理 README / GitHub 展示时应注明。

---

# Localization 阶段补充结果

## 1. 任务说明
Localization 为论文中的单任务监督学习 benchmark，论文主表报告其为 6 分类任务。

## 2. 当前复现结果（Transformer, test_id）
- Test Accuracy: 0.9927
- Test F1-score: 0.9927
- Best epoch: 63
- Validation Accuracy: 0.9944

## 3. 与论文对照
论文中 Transformer 在 Localization 上的主表结果为：
- Accuracy: 99.27
- F1-score: 99.27

当前复现结果与论文主表基本一致，可视为高保真 baseline 复现。

## 4. 已知问题
Localization 的 `test_easy_id` 与 `test_medium_id` 当前加载后为空，导致若直接使用 `--test_splits="all"` 会在评估阶段触发除零错误。
因此当前结果基于 `--test_splits="test_id"` 获得，属于主测试集有效复现，difficulty split 结果暂未完整补齐。
