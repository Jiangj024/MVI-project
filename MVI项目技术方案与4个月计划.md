# MVI微血管侵犯预测项目 - 技术方案与4个月计划

**项目定位**: 硕士毕业大论文 + 算法类专利  
**时间安排**: 4个月（专利）+ 半年（大论文）  
**目标**: 方法创新、AI方向、保证工作量  
**创建时间**: 2026-05-03

---

## 目录

1. [项目背景与现状分析](#1-项目背景与现状分析)
2. [核心创新点设计](#2-核心创新点设计)
3. [技术架构详细设计](#3-技术架构详细设计)
4. [实验设计方案](#4-实验设计方案)
5. [4个月半月计划](#5-4个月半月计划)
6. [论文与专利策略](#6-论文与专利策略)
7. [风险评估与应对](#7-风险评估与应对)
8. [附录：代码实现指南](#8-附录代码实现指南)

---

## 1. 项目背景与现状分析

### 1.1 数据现状

#### **现有数据（201例）**
- **MVI阳性**: 51例（临床数据）+ 181张影像
- **MVI阴性**: 150例（临床数据）+ 209张影像
- **样本不平衡**: 约3:1（阴性:阳性）
- **总图像数**: 796张

#### **数据类型**
1. **多时相超声造影图像**:
   - grey（灰阶）
   - ap（动脉期）- 高增强
   - pp（门脉期）- 开始消退
   - lp（延迟期）- 低增强

2. **临床特征（22维）**:
   - 基本信息：性别、年龄
   - 病毒感染：HBV、HCV
   - 肝功能：总胆红素、直接胆红素、总蛋白、白蛋白、ALT、AST、碱性磷酸酶、谷氨酰转移酶、总胆酸、乳酸脱氢酶
   - 肿瘤标志物：甲胎蛋白、癌胚抗原、CA199、CA125、甲胎异质体、异常凝血酶原、凝血酶原时间

#### **关键临床指标差异**
| 指标 | MVI+ 均值 | MVI- 均值 | 差异显著性 |
|------|-----------|-----------|------------|
| 甲胎蛋白 | 4390.28 | 1354.45 | ⭐⭐⭐ |
| 异常凝血酶原 | 8044.35 | 885.20 | ⭐⭐⭐ |
| 总胆红素 | 12.22 | 12.90 | - |
| 白蛋白 | 41.84 | 42.12 | - |

#### **新增数据（预计半个月内）**
- **数量**: 约200例
- **总数据量**: 约400例（数据量翻倍）

#### **标注质量**
- ✅ 临床医生标注的矩形框
- ✅ 标注质量可靠

---

### 1.2 现有模型表现

#### **方案A：MultiModal-ViT（5维核心特征）**
```
特征：甲胎蛋白、异常凝血酶原、HBV、年龄、性别

性能指标：
- Mean AUC: 0.7619
- Mean ACC: 0.72
- Mean SEN: 0.6900 ⚠️
- Mean SPE: 0.6923 ⚠️

评价：
✅ 极致均衡
✅ 抗漏诊能力最强
✅ 最具临床落地价值
❌ 敏感性和特异性偏低
```

#### **方案B：MultiModal-ViT（21维全量特征）**
```
特征：全部22维临床指标

性能指标：
- Mean AUC: 0.7667
- Mean ACC: 0.72
- Mean SEN: 0.6142 ⚠️⚠️
- Mean SPE: 0.7770

评价：
✅ 理论天花板最高
✅ 整体排序能力强
❌ 敏感性更低（0.6142）
❌ 一般性肝功能指标稀释了肿瘤特异性信号
```

#### **已知的技术经验**
1. ✅ **ViT-B/16 > ResNet18**: 多模态融合效果更好（AUC差距0.035）
2. ✅ **禁止MixUp**: 会破坏临床数据的物理意义
3. ✅ **禁止RandomVerticalFlip**: 会破坏超声图像的近远场深度
4. ✅ **最优损失函数**: CrossEntropyLoss + class_weights=[1.0, 3.0] + label_smoothing=0.1
5. ❌ **WeightedRandomSampler失败**: 导致少数类过拟合（SEN=0.846, SPE=0.469）
6. ❌ **Focal Loss失败**: 导致模型过度保守（SEN暴跌）

---

### 1.3 存在的问题

#### **问题1：敏感性和特异性偏低** ⚠️⚠️⚠️
```
现状：
- SEN: 0.69（方案A）/ 0.61（方案B）
- SPE: 0.69（方案A）/ 0.78（方案B）

临床影响：
- 低SEN → 漏诊率高（31%的MVI+患者被误判为MVI-）
- 低SPE → 误诊率高（31%的MVI-患者被误判为MVI+）

根本原因：
1. 样本不平衡（3:1）导致模型偏向阴性类
2. 现有损失函数（class_weights=[1.0, 3.0]）不够精细
3. 难样本（边界病例）未得到充分关注
```

#### **问题2：未充分利用多时相信息** ⚠️⚠️
```
现状：
- 现有模型将多时相图像视为独立样本
- 未建模时序演变模式（动脉期高增强 → 延迟期低增强）

潜在提升空间：
- 时序建模可以捕捉增强-消退模式
- 不同时相的重要性可能不同（需要自适应权重）
```

#### **问题3：影像-临床特征融合不够深入** ⚠️
```
现状：
- Late Fusion（简单拼接）
- 影像特征和临床特征缺乏交互

潜在提升空间：
- Cross-Attention可以让影像特征查询临床特征
- 门控融合可以动态调整两种模态的权重
```

---

### 1.4 代码审查发现的关键问题（2026-05-03 补充）

> 以下问题在审查现有代码（`vit2.py`, `resnet18.py`, `0318vit2.py`）后发现，
> 部分问题直接影响当前实验结果的可信度，必须在开展任何创新点之前修复。

#### **问题A：数据泄露 — KFold 按图像切分而非患者切分** 🚨🚨🚨

```
严重程度：致命
影响范围：所有已报告的实验结果（AUC=0.76）不可信

现状：
- KFold 直接对图像索引切分：kf.split(base_dataset)
- 同一患者有 4-8 张图像（4个时相 × 1-2种造影剂 sonovue/sonazoid）
- 同一患者的 ap 可能在训练集，pp 可能在测试集
- 模型在训练时已见过该患者的影像风格和临床特征

后果：
- 当前 AUC=0.76 存在虚高，真实性能可能在 0.65-0.72
- 所有基于该 baseline 的创新点预期提升需要重新评估

修复方案：
- 先提取所有 patient_id，对 patient_id 做 KFold
- 同一患者的所有图像必须划入同一 fold
- 修复后重新跑 baseline，获取真实性能基准
```

#### **问题B：临床特征未做标准化** 🚨🚨

```
严重程度：高
影响范围：多模态融合效果受限

现状：
- 仅年龄做了 Z-score 标准化
- 甲胎蛋白均值 4390，异常凝血酶原均值 8044
- 性别 0/1，HBV 0/1
- 数量级差距达 4 个量级

后果：
- MLP 梯度被大数值特征（AFP、凝血酶原）主导
- 小数值特征（性别、HBV）几乎无学习信号
- 临床分支实质上退化为仅依赖 AFP 和凝血酶原

修复方案：
- 二分类特征（性别、HBV、HCV）保持 0/1 不变
- 所有连续型特征统一做 Z-score 标准化
- 缺失值改用中位数填充（fillna(0) 对 AFP 等有误导）
```

#### **问题C：当前代码没有多时相建模** 🚨

```
严重程度：中（不影响当前结果，但影响创新点实现）

现状：
- MultimodalDataset 每条数据是一张独立图像
- 模型不知道哪几张图属于同一患者
- 模型不知道图像是 grey 还是 ap
- "多时相图像视为独立样本" 不是设计选择，而是 Dataset 没实现

前置工作：
- 需要新建 MultiPhaseDataset 类
- 每条数据 = 一个患者的 4 个时相图像
- 需要处理缺失时相和多造影剂的情况
```

#### **问题D：WeightedRandomSampler 与损失函数配置矛盾** ⚠️

```
现状：
- vit2.py 同时使用了 WeightedRandomSampler（平衡采样）
  和 class_weights=[1.0, 1.0]（均等权重损失）
- 文档中总结的最优配置是 class_weights=[1.0, 3.0]，代码未体现
- 文档中总结 WeightedRandomSampler 会导致过拟合（SEN=0.846, SPE=0.469），
  但代码仍在使用

修复方案：
- 去掉 WeightedRandomSampler，改回 shuffle=True
- 使用 class_weights=[1.0, 3.0] 的 CrossEntropyLoss
```

#### **问题E：图像文件命名格式补充说明**

```
实际文件命名格式（此前文档未记录）：

  {患者ID}_{时相}_{造影剂}_crop.jpg   # 造影增强时相
  {患者ID}_grey_crop.jpg              # 灰阶基线

示例：
  ZS09010040_ap_sonovue_crop.jpg      # sonovue 造影剂
  ZS09222749_ap_sonazoid_crop.jpg     # sonazoid 造影剂（不同造影剂）
  ZS09010040_grey_crop.jpg            # 灰阶基线

关键发现：
- 同一患者同一时相可能有多张图（不同造影剂）
- 实际每个患者可能有 4-8 张图像，而非固定 4 张
- 造影剂类型（sonovue vs sonazoid）可能影响增强模式
  → 后续可以作为额外特征或单独建模
```

---

### 1.5 改进目标（原1.4，目标值待修正后 baseline 确定后更新）

#### **性能目标（修正数据泄露前后对比）**

> ⚠️ 修复患者级别 KFold 后，baseline AUC 预计会下降。
> 下表中"修正后 baseline"为预估值，需要实际跑出来后更新。

| 指标 | 原 baseline（含泄露） | 修正后 baseline（预估） | 最终目标 | 相对修正 baseline 提升 |
|------|----------------------|------------------------|----------|----------------------|
| AUC | 0.76 | **0.65-0.72** | **0.78+** | +0.06~0.13 |
| ACC | 0.72 | **0.65-0.70** | **0.75+** | +0.05~0.10 |
| SEN | 0.69 | **0.55-0.65** | **0.72+** | +0.07~0.17 |
| SPE | 0.69 | **0.65-0.72** | **0.75+** | +0.03~0.10 |

**说明**: 修正后 baseline 如果低于 0.65，说明现有模型在无泄露条件下学到的
信号很弱，此时创新点的提升空间反而更大。等 200 例新数据到位后（总量 ~400 例），
仅靠数据量翻倍就可能带来 AUC +0.03~0.05 的提升。

#### **创新目标**
1. ✅ **3个主创新点** + 2个辅助创新点
2. ✅ 足够支撑硕士大论文（8-10万字）
3. ✅ 足够支撑算法类专利申请
4. ✅ 足够发表1篇小论文（可选）

#### **工程目标**
1. ✅ 训练时间 < 30分钟（多卡训练）
2. ✅ 代码清晰易懂（自己能看明白）
3. ✅ 实验记录完整（TensorBoard + 表格）
4. ✅ 可复现性强（固定随机种子）

---

## 2. 核心创新点设计

### 2.1 创新点概览

```
┌─────────────────────────────────────────────────────────────┐
│                    MVI预测模型架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  主创新点1: 多时相时序Transformer                    │  │
│  │  ├─ 时序自注意力机制                                 │  │
│  │  ├─ 自适应时相权重学习                               │  │
│  │  └─ 时序对比学习                                     │  │
│  │  目标: AUC 0.76 → 0.82+                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  主创新点2: 跨模态动态融合                          │  │
│  │  ├─ Cross-Attention（影像查询临床）                 │  │
│  │  ├─ 门控融合机制                                     │  │
│  │  └─ 特征重要性可视化                                 │  │
│  │  目标: 提升特征交互能力                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  主创新点3: 类别不平衡优化策略                      │  │
│  │  ├─ 改进的Focal Loss                                │  │
│  │  ├─ 难样本挖掘                                       │  │
│  │  └─ 阈值优化（Youden指数）                          │  │
│  │  目标: SEN/SPE 0.69 → 0.75+                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  辅助创新点4: 连续值特征智能处理                    │  │
│  │  ├─ 特征选择（LASSO/SHAP）                          │  │
│  │  ├─ 自适应特征加权                                   │  │
│  │  └─ 特征交互建模                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  辅助创新点5: 不确定性量化                          │  │
│  │  ├─ Monte Carlo Dropout                             │  │
│  │  ├─ 集成学习                                         │  │
│  │  └─ 置信度输出                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### **创新点优先级**
| 创新点 | 优先级 | 工作量 | 预期提升 | 论文价值 | 专利价值 |
|--------|--------|--------|----------|----------|----------|
| 多时相时序Transformer | ⭐⭐⭐ | 高 | AUC +0.04 | 高 | 高 |
| 跨模态动态融合 | ⭐⭐⭐ | 中 | AUC +0.02 | 高 | 高 |
| 类别不平衡优化 | ⭐⭐⭐ | 中 | SEN/SPE +0.06 | 中 | 高 |
| 连续值特征处理 | ⭐⭐ | 低 | AUC +0.01 | 中 | 中 |
| 不确定性量化 | ⭐ | 低 | 可解释性 | 中 | 低 |

---

### 2.2 主创新点1：多时相时序Transformer

#### **核心思想**
利用Transformer的自注意力机制，建模多时相超声造影图像的时序演变模式（动脉期高增强 → 门脉期消退 → 延迟期低增强），自动学习不同时相的重要性权重。

#### **技术细节**

##### **2.2.1 时序编码**
```python
# 时相顺序编码
phase_order = {
    'grey': 0,  # 基线
    'ap': 1,    # 动脉期（高增强）
    'pp': 2,    # 门脉期（开始消退）
    'lp': 3     # 延迟期（低增强）
}

# 位置编码（Sinusoidal Positional Encoding）
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

##### **2.2.2 时序Transformer架构**
```
输入: [grey, ap, pp, lp] 四个时相的ViT特征
      每个时相: [B, 768] (ViT-B/16输出)

步骤1: 时相嵌入
  - 添加位置编码: [B, 4, 768]
  - 添加可学习的时相类型嵌入

步骤2: 时序自注意力
  - Multi-Head Self-Attention (8 heads)
  - 捕捉时相间的依赖关系
  - 学习增强-消退模式

步骤3: 时相权重学习
  - 自适应权重: α = softmax(W·h + b)
  - 加权融合: h_fused = Σ(α_i · h_i)

输出: [B, 768] 融合后的时序特征
```

##### **2.2.3 时序对比学习（修正版：监督对比学习）**

> **设计修正说明**: 原方案拉近同一患者不同时相的特征，但 MVI 的判别信号恰恰
> 在于时相间的差异模式（"快进快出" vs "均匀增强"），强行拉近会抹平关键信号。
> 改为监督对比学习（Supervised Contrastive Learning），直接服务于分类目标。

```python
# 修正后的对比学习目标：拉近同一 MVI 标签的患者融合特征，推远不同标签的

正样本对：
- (h_fused_i, h_fused_j): 标签相同的两个患者的融合特征 (y_i == y_j)

负样本对：
- (h_fused_i, h_fused_k): 标签不同的两个患者的融合特征 (y_i != y_k)

损失函数（SupCon Loss）：
L_supcon = Σ_i  -1/|P(i)| Σ_{p∈P(i)} log(
    exp(sim(z_i, z_p) / τ) / Σ_{a≠i} exp(sim(z_i, z_a) / τ)
)

其中：
- z_i = projection_head(h_fused_i)  # 投影后的特征
- P(i): 与样本 i 同标签的所有样本集合
- τ: 温度参数（0.1）
- sim(·,·): 余弦相似度

优势：
- 直接优化类间可分性
- 不会破坏时相间的差异模式
- 与分类损失互补：分类损失学决策边界，对比损失学特征空间结构
```

#### **预期效果**
- ✅ AUC提升: 0.76 → 0.80 (+0.04)
- ✅ 捕捉时序演变模式
- ✅ 自动学习时相重要性
- ✅ 可解释性强（可视化时相权重）

#### **实现难度**
- 难度: ⭐⭐⭐ (中等)
- 训练时间: 约20分钟（多卡训练）
- 代码量: 约300行

---

### 2.3 主创新点2：跨模态动态融合

#### **核心思想**
使用Cross-Attention机制让影像特征主动查询临床特征，实现两种模态的深度交互，而不是简单的Late Fusion拼接。

#### **技术细节**

##### **2.3.1 Cross-Attention机制（修正版：逐指标 token 设计）**

> **设计修正说明**: 原方案将临床特征先经 MLP 压缩到 [B, 64] 再做 K/V，
> 此时 softmax 注意力只有一个 token，权重恒等于 1，退化为线性投影+残差。
> 修正为：将 22 个临床指标拆成 22 个 token，每个 token 对应一个指标，
> 这样 attention map 为 [B, 1, 22]，可以真正学到影像在查询哪些指标。

```
输入:
- 影像特征 (Query): [B, 1, D]  D=768, 来自时序Transformer
- 临床特征 (Key/Value): [B, 22, D_c]  每个指标独立嵌入
  D_c = 32（每个标量指标通过 nn.Linear(1, 32) 嵌入）

步骤1: 指标嵌入
  每个临床指标 x_i ∈ R → 嵌入到 d_c 维
  h_clinical = [embed(x_1), embed(x_2), ..., embed(x_22)]  [B, 22, 32]
  可选：加上可学习的指标类型嵌入（类似 token type embedding）

步骤2: 线性投影
  Q = W_q · h_image       [B, 1, d_attn]    d_attn=64
  K = W_k · h_clinical    [B, 22, d_attn]
  V = W_v · h_clinical    [B, 22, d_attn]

步骤3: 注意力计算
  scores = QK^T / √d_attn            [B, 1, 22]
  α = softmax(scores, dim=-1)        [B, 1, 22]  ← 每个临床指标的注意力权重
  attended = α · V                    [B, 1, d_attn]

步骤4: 输出投影 + 残差连接
  output = W_o · attended.squeeze(1)  [B, 768]
  h_fused = h_image + output

输出: [B, 768] 融合后的特征

可视化价值:
  α ∈ [B, 1, 22] 直接展示模型关注哪些临床指标
  → 可做热图：MVI+ vs MVI- 患者的注意力差异
  → 可做柱状图：Top-K 重要指标排名
```

##### **2.3.2 门控融合机制**
```python
# 动态调整影像和临床特征的权重

门控单元:
  g = σ(W_g · [h_image; h_clinical] + b_g)  # [B, 1]
  
融合:
  h_fused = g · h_image + (1-g) · h_clinical_proj
  
其中:
- σ: Sigmoid激活函数
- g ∈ [0, 1]: 门控权重
- h_clinical_proj: 临床特征投影到768维
```

##### **2.3.3 特征重要性可视化**
```python
# 使用注意力权重可视化哪些临床特征对预测贡献最大

注意力权重: α = softmax(QK^T / √d_k)  # [B, 1, 22]

可视化:
- 热图: 显示每个临床特征的注意力权重
- 柱状图: Top-K重要特征
- 散点图: 特征值 vs 注意力权重
```

#### **预期效果**
- ✅ AUC提升: 0.80 → 0.82 (+0.02)
- ✅ 影像-临床特征深度交互
- ✅ 可解释性强（可视化重要特征）
- ✅ 动态调整模态权重

#### **实现难度**
- 难度: ⭐⭐ (较低)
- 训练时间: 约5分钟（增量）
- 代码量: 约150行

---

### 2.4 主创新点3：类别不平衡优化策略

#### **核心思想**
针对3:1的样本不平衡问题，设计改进的Focal Loss和难样本挖掘策略，重点提升敏感性和特异性。

#### **技术细节**

##### **2.4.1 改进的Focal Loss（多模态版）**
```python
# 原始Focal Loss在多模态下过度保守，需要改进

原始Focal Loss:
  FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
  
改进版（考虑临床特征的置信度）:
  FL_mm(p_t, c) = -α_t · (1 - p_t)^γ · β(c) · log(p_t)
  
其中:
- p_t: 预测概率
- α_t: 类别权重（阳性3.0，阴性1.0）
- γ: 聚焦参数（2.0）
- β(c): 临床置信度调制因子
  β(c) = 1 + λ · |c - c_mean|  # 临床特征偏离均值越大，权重越高
  
参数设置:
- α_阳性 = 3.0（重点关注阳性样本）
- α_阴性 = 1.0
- γ = 2.0（标准设置）
- λ = 0.5（临床调制强度）
```

##### **2.4.2 难样本挖掘（Hard Example Mining）**
```python
# 识别并重点训练难分样本

难样本定义:
1. 预测概率接近0.5的样本（边界样本）
2. 预测错误的样本（误分样本）
3. 临床特征矛盾的样本（如AFP正常但MVI+）

挖掘策略:
  hard_mask = (0.3 < p < 0.7) | (pred != label)
  loss_hard = loss[hard_mask].mean()
  loss_all = loss.mean()
  loss_total = 0.7 · loss_all + 0.3 · loss_hard

效果:
- 强制模型关注难分样本
- 提升边界样本的判别能力
```

##### **2.4.3 阈值优化（Youden指数）**
```python
# 不使用默认0.5阈值，根据Youden指数优化

Youden指数:
  J = SEN + SPE - 1
  
最优阈值:
  threshold* = argmax_t (SEN(t) + SPE(t) - 1)
  
实现:
1. 在验证集上遍历阈值 [0.1, 0.9]，步长0.01
2. 计算每个阈值的SEN和SPE
3. 选择Youden指数最大的阈值
4. 在测试集上使用该阈值

预期:
- 平衡SEN和SPE
- 根据临床需求调整（如优先降低漏诊率）
```

#### **预期效果**
- ✅ SEN提升: 0.69 → 0.75+ (+0.06)
- ✅ SPE提升: 0.69 → 0.75+ (+0.06)
- ✅ 平衡敏感性和特异性
- ✅ 降低漏诊率和误诊率

#### **实现难度**
- 难度: ⭐⭐ (较低)
- 训练时间: 约5分钟（增量）
- 代码量: 约200行

---

### 2.5 辅助创新点4：连续值特征智能处理

#### **核心思想**
保留临床特征的连续值（不二分类），使用特征选择和自适应加权，充分利��临床信息。

#### **技术细节**

##### **2.5.1 特征选择（LASSO + SHAP）**
```python
# 从22维特征中筛选最重要的特征

方法1: LASSO回归
  - 使用L1正则化自动筛选特征
  - 保留系数非零的特征
  
方法2: SHAP值
  - 计算每个特征的SHAP值
  - 保留SHAP值Top-K的特征
  
组合策略:
  - 取两种方法的交集
  - 预期筛选出8-12个关键特征
```

##### **2.5.2 自适应特征加权**
```python
# 学习每个特征的重要性权重

特征加权模块:
  w = softmax(W · h_clinical + b)  # [B, K]
  h_weighted = w ⊙ h_clinical  # 逐元素相乘
  
效果:
- 自动学习特征重要性
- 不同样本的特征权重可能不同
```

##### **2.5.3 特征交互建模**
```python
# 建模特征之间的交互（如AFP × 异常凝血酶原）

交互项:
  h_interact = [h_i · h_j for i, j in feature_pairs]
  
特征对选择:
- (甲胎蛋白, 异常凝血酶原)
- (HBV, 年龄)
- (白蛋白, 总胆红素)
```

#### **预期效果**
- ✅ AUC提升: 0.82 → 0.83 (+0.01)
- ✅ 充分利用临床信息
- ✅ 可解释性强（特征重要性）

#### **实现难度**
- 难度: ⭐ (低)
- 训练时间: 约3分钟（增量）
- 代码量: 约100行

---

### 2.6 辅助创新点5：不确定性量化

#### **核心思想**
输出预测的置信度，帮助临床医生判断预测的可靠性。

#### **技术细节**

##### **2.6.1 Monte Carlo Dropout**
```python
# 在测试时保持Dropout开启，多次前向传播

预测过程:
1. 对同一样本进行T次前向传播（T=20）
2. 得到T个预测概率: [p_1, p_2, ..., p_T]
3. 计算均值和标准差:
   - 预测概率: p_mean = mean([p_1, ..., p_T])
   - 不确定性: p_std = std([p_1, ..., p_T])

置信度:
  confidence = 1 - p_std
  
解释:
- p_std小 → 置信度高 → 预测可靠
- p_std大 → 置信度低 → 建议进一步检查
```

##### **2.6.2 集成学习**
```python
# 5折交叉验证的5个模型进行集成

集成策略:
  p_ensemble = mean([p_fold1, p_fold2, ..., p_fold5])
  
不确定性:
  uncertainty = std([p_fold1, p_fold2, ..., p_fold5])
```

##### **2.6.3 置信度输出**
```python
# 输出格式

预测结果:
{
  "prediction": "MVI+",
  "probability": 0.85,
  "confidence": 0.92,
  "uncertainty": 0.08,
  "recommendation": "高置信度预测，可直接采纳"
}

置信度分级:
- 高置信度 (>0.9): 可直接采纳
- 中置信度 (0.7-0.9): 建议结合其他检查
- 低置信度 (<0.7): 建议进一步检查
```

#### **预期效果**
- ✅ 提供预测置信度
- ✅ 辅助临床决策
- ✅ 提升模型可信度

#### **实现难度**
- 难度: ⭐ (低)
- 训练时间: 0分钟（测试时计算）
- 代码量: 约50行

---

## 3. 技术架构详细设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        输入数据                                      │
│  ┌──────────────────────┐        ┌──────────────────────┐          │
│  │  多时相CEUS图像      │        │  临床特征（22维）    │          │
│  │  [grey, ap, pp, lp]  │        │  连续值 + 分类值     │          │
│  └──────────────────────┘        └──────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                    ↓                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    特征提取层                                        │
│  ┌──────────────────────┐        ┌──────────────────────┐          │
│  │  ViT-B/16 (预训练)   │        │  临床特征MLP         │          │
│  │  每个时相 → 768维    │        │  22维 → 64维         │          │
│  │  [B, 4, 768]         │        │  [B, 64]             │          │
│  └──────────────────────┘        └──────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│              创新点1: 多时相时序Transformer                          │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  时序编码 + 自注意力 + 时相权重学习                  │          │
│  │  [B, 4, 768] → [B, 768]                              │          │
│  └──────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                    ↓                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│              创新点2: 跨模态动态融合                                 │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Cross-Attention + 门控融合                          │          │
│  │  影像[B,768] × 临床[B,64] → [B, 832]                 │          │
│  └──────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    分类器                                            │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  MLP: 832 → 256 → 128 → 2                            │          │
│  │  Dropout + BatchNorm + ReLU                          │          │
│  └──────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│              创新点3: 类别不平衡优化                                 │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  改进Focal Loss + 难样本挖掘 + 阈值优化              │          │
│  └──────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    输出                                              │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  预测类别: MVI+ / MVI-                                │          │
│  │  预测概率: [0, 1]                                     │          │
│  │  置信度: [0, 1] (创新点5)                            │          │
│  └──────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

#### **模型参数量（修正：ViT 权重共享）**

> **修正说明**: 4个时相应共享同一个 ViT backbone（权重共享），而非 4 个独立 ViT。
> 理由：(1) 不同时相本质上是同一超声图的不同增强阶段，底层视觉特征相似；
> (2) 201例患者的数据量不足以训练 4 个独立 ViT；(3) 共享权重是参数高效的标准做法。

| 模块 | 参数量 | 是否冻结 |
|------|--------|----------|
| ViT-B/16 (共享，4时相复用) | 86M | 部分冻结（仅解冻最后1层，约7M可训练） |
| 时序Transformer | 2M | 可训练 |
| 临床指标嵌入 + MLP | 0.2M | 可训练 |
| 跨模态融合 | 0.5M | 可训练 |
| 分类器 | 0.3M | 可训练 |
| **总计** | **~89M** | **可训练: ~10M** |

#### **显存占用估算（修正）**
```
单卡（Batch Size = 16）:
- 模型参数（共享ViT）: 89M × 4 bytes = 0.36GB
- 激活值（4个时相前向传播）: 约4GB（每个时相 ~1GB）
- 梯度（可训练部分）: 10M × 4 bytes = 40MB
- 优化器状态: 10M × 8 bytes = 80MB
- 总计: 约4.5GB

注意：4个时相复用同一 ViT，但前向传播需要 4 次（串行），
激活值需要保留4份用于反向传播。可用 gradient checkpointing 减少到 ~2.5GB。
```

---

### 3.2 多时相时序模块

#### **3.2.1 模块架构**

```python
class TemporalTransformer(nn.Module):
    """
    多时相时序Transformer模块
    输入: [B, 4, 768] (4个时相的ViT特征)
    输出: [B, 768] (融合后的时序特征)
    """
    def __init__(self, d_model=768, nhead=8, num_layers=2):
        super().__init__()
        
        # 1. 时相位置编码
        self.phase_embedding = nn.Embedding(4, d_model)  # 4个时相
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 2. 时序Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 3. 时相权重学习
        self.phase_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
        
        # 4. 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 128)
        )
    
    def forward(self, phase_features, return_attention=False):
        """
        Args:
            phase_features: [B, 4, 768] 四个时相的特征
            return_attention: 是否返回注意力权重
        Returns:
            fused_features: [B, 768] 融合后的特征
            attention_weights: [B, 4] 时相权重（可选）
        """
        B, N, D = phase_features.shape  # [B, 4, 768]
        
        # 1. 添加位置编码
        phase_ids = torch.arange(N, device=phase_features.device)
        phase_emb = self.phase_embedding(phase_ids)  # [4, 768]
        x = phase_features + phase_emb.unsqueeze(0)  # [B, 4, 768]
        
        # 2. Transformer编码
        x = x.transpose(0, 1)  # [4, B, 768] (Transformer要求)
        x = self.transformer_encoder(x)  # [4, B, 768]
        x = x.transpose(0, 1)  # [B, 4, 768]
        
        # 3. 时相权重学习
        attention_logits = self.phase_attention(x)  # [B, 4, 1]
        attention_weights = F.softmax(attention_logits.squeeze(-1), dim=1)  # [B, 4]
        
        # 4. 加权融合
        fused_features = torch.sum(
            x * attention_weights.unsqueeze(-1),  # [B, 4, 768]
            dim=1
        )  # [B, 768]
        
        if return_attention:
            return fused_features, attention_weights
        return fused_features
    
    def contrastive_loss(self, fused_features, labels, temperature=0.1):
        """
        监督对比学习损失（修正版）
        
        修正说明：
        原版拉近同一患者不同时相的特征，会抹平时相间差异模式。
        修正为 Supervised Contrastive Loss：
        - 同标签患者的融合特征互为正样本
        - 不同标签患者的融合特征互为负样本
        
        Args:
            fused_features: [B, 768] 时序Transformer输出的融合特征
            labels: [B] MVI标签 (0 or 1)
            temperature: 温度参数
        Returns:
            loss: SupCon损失
        """
        B = fused_features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=fused_features.device)
        
        # 投影到对比学习空间
        z = self.projection_head(fused_features)  # [B, 128]
        z = F.normalize(z, dim=1)
        
        # 相似度矩阵
        sim_matrix = torch.matmul(z, z.T) / temperature  # [B, B]
        
        # 标签掩码：同标签为正样本
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float()  # [B, B]
        mask_pos.fill_diagonal_(0)  # 排除自身
        
        # 排除自身的 log-sum-exp
        mask_self = torch.eye(B, device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask_self, -1e9)
        
        # 对每个样本计算 SupCon loss
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        
        # 只对有正样本的求平均
        num_pos = mask_pos.sum(dim=1)
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-6)
        
        # 排除没有正样本的（极端情况：某类只有1个样本）
        valid = num_pos > 0
        loss = -mean_log_prob_pos[valid].mean()
        
        return loss
```

#### **3.2.2 位置编码实现**

```python
class PositionalEncoding(nn.Module):
    """
    Sinusoidal位置编码
    """
    def __init__(self, d_model, max_len=10):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D]
        Returns:
            x + pe: [B, N, D]
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)
```

---

### 3.3 跨模态融合模块

#### **3.3.1 Cross-Attention实现**

```python
class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合模块（修正版：逐指标 token）
    
    修正说明：
    原版将临床特征压缩为 [B,1,256] 的单 token 做 K/V，
    softmax 后注意力恒为 1，退化为线性层。
    修正为：22 个临床指标各自嵌入为独立 token，
    attention map [B,1,22] 可以学到影像在查询哪些指标。
    """
    def __init__(self, image_dim=768, num_indicators=22, indicator_embed_dim=32, attn_dim=64):
        super().__init__()
        
        # 每个临床指标: 标量 → indicator_embed_dim 维
        self.indicator_embed = nn.Linear(1, indicator_embed_dim)
        # 可学习的指标类型嵌入（区分 AFP / 凝血酶原 / 年龄 等）
        self.indicator_type_embed = nn.Embedding(num_indicators, indicator_embed_dim)
        
        # Query: 影像特征
        self.query_proj = nn.Linear(image_dim, attn_dim)
        # Key/Value: 指标嵌入
        self.key_proj = nn.Linear(indicator_embed_dim, attn_dim)
        self.value_proj = nn.Linear(indicator_embed_dim, attn_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(attn_dim, image_dim)
        
        self.scale = attn_dim ** -0.5
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(image_dim)
        self.num_indicators = num_indicators
    
    def forward(self, image_features, clinical_raw, return_attention=False):
        """
        Args:
            image_features: [B, 768] 影像特征（来自时序Transformer）
            clinical_raw: [B, 22] 原始临床特征值（标准化后）
            return_attention: 是否返回注意力权重
        Returns:
            fused_features: [B, 768]
            attention_weights: [B, 22] 每个临床指标的注意力权重（可选）
        """
        B = image_features.shape[0]
        
        # 1. 指标嵌入: [B, 22] → [B, 22, 1] → [B, 22, embed_dim]
        indicator_embeds = self.indicator_embed(clinical_raw.unsqueeze(-1))  # [B, 22, 32]
        # 加上指标类型嵌入
        type_ids = torch.arange(self.num_indicators, device=clinical_raw.device)
        indicator_embeds = indicator_embeds + self.indicator_type_embed(type_ids)  # [B, 22, 32]
        
        # 2. 投影
        Q = self.query_proj(image_features).unsqueeze(1)   # [B, 1, attn_dim]
        K = self.key_proj(indicator_embeds)                 # [B, 22, attn_dim]
        V = self.value_proj(indicator_embeds)                # [B, 22, attn_dim]
        
        # 3. 注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, 1, 22]
        attn_weights = F.softmax(scores, dim=-1)                      # [B, 1, 22]
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.matmul(attn_weights, V)  # [B, 1, attn_dim]
        attended = attended.squeeze(1)              # [B, attn_dim]
        
        # 4. 输出投影 + 残差 + LayerNorm
        output = self.out_proj(attended)            # [B, 768]
        fused_features = self.layer_norm(image_features + output)
        
        if return_attention:
            return fused_features, attn_weights.squeeze(1)  # [B, 22]
        return fused_features
```

#### **3.3.2 门控融合实现**

```python
class GatedFusion(nn.Module):
    """
    门控融合模块
    动态调整影像和临床特征的权重
    """
    def __init__(self, image_dim=768, clinical_dim=64):
        super().__init__()
        
        # 临床特征投影到影像维度
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, image_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 门控单元
        self.gate = nn.Sequential(
            nn.Linear(image_dim + clinical_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image_features, clinical_features):
        """
        Args:
            image_features: [B, 768]
            clinical_features: [B, 64]
        Returns:
            fused_features: [B, 768]
        """
        # 临床特征投影
        clinical_proj = self.clinical_proj(clinical_features)  # [B, 768]
        
        # 计算门控权重
        concat_features = torch.cat([image_features, clinical_features], dim=1)  # [B, 832]
        gate_weight = self.gate(concat_features)  # [B, 1]
        
        # 门控融合
        fused_features = gate_weight * image_features + (1 - gate_weight) * clinical_proj
        
        return fused_features, gate_weight
```

---

### 3.4 类别不平衡处理模块

#### **3.4.1 改进的Focal Loss**

```python
class ImprovedFocalLoss(nn.Module):
    """
    改进的Focal Loss（多模态版）
    考虑临床特征的置信度
    """
    def __init__(self, alpha=3.0, gamma=2.0, lambda_clinical=0.5):
        super().__init__()
        self.alpha = alpha  # 阳性类权重
        self.gamma = gamma  # 聚焦参数
        self.lambda_clinical = lambda_clinical  # 临床调制强度
    
    def forward(self, logits, labels, clinical_features):
        """
        Args:
            logits: [B, 2] 模型输出
            labels: [B] 标签
            clinical_features: [B, 64] 临床特征
        Returns:
            loss: Focal Loss
        """
        # 预测概率
        probs = F.softmax(logits, dim=1)  # [B, 2]
        pt = probs[range(len(labels)), labels]  # [B] 正确类别的概率
        
        # 类别权重
        alpha_t = torch.where(labels == 1, self.alpha, 1.0)  # [B]
        
        # 临床置信度调制
        # 计算临床特征偏离均值的程度
        clinical_mean = clinical_features.mean(dim=0, keepdim=True)  # [1, 64]
        clinical_deviation = torch.abs(clinical_features - clinical_mean).mean(dim=1)  # [B]
        beta = 1 + self.lambda_clinical * clinical_deviation  # [B]
        
        # Focal Loss
        focal_weight = alpha_t * beta * (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # [B]
        loss = (focal_weight * ce_loss).mean()
        
        return loss
```

#### **3.4.2 难样本挖掘**

```python
class HardExampleMining:
    """
    难样本挖掘策略
    """
    def __init__(self, hard_ratio=0.3):
        self.hard_ratio = hard_ratio
    
    def __call__(self, logits, labels, loss_per_sample):
        """
        Args:
            logits: [B, 2]
            labels: [B]
            loss_per_sample: [B] 每个样本的损失
        Returns:
            loss: 加权后的损失
        """
        # 预测概率
        probs = F.softmax(logits, dim=1)[:, 1]  # [B] 阳性概率
        preds = (probs > 0.5).long()  # [B]
        
        # 识别难样本
        # 1. 边界样本（概率接近0.5）
        boundary_mask = (probs > 0.3) & (probs < 0.7)
        
        # 2. 误分样本
        error_mask = (preds != labels)
        
        # 3. 难样本mask
        hard_mask = boundary_mask | error_mask
        
        # 计算损失
        loss_all = loss_per_sample.mean()
        loss_hard = loss_per_sample[hard_mask].mean() if hard_mask.sum() > 0 else 0.0
        
        # 加权组合
        loss = (1 - self.hard_ratio) * loss_all + self.hard_ratio * loss_hard
        
        return loss
```

---

### 3.5 训练策略

#### **3.5.1 优化器配置**

```python
def configure_optimizers(model, lr=1e-4, weight_decay=0.05):
    """
    配置优化器：分层学习率
    """
    # 参数分组
    vit_params = []
    temporal_params = []
    fusion_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'vit' in name:
            vit_params.append(param)
        elif 'temporal' in name:
            temporal_params.append(param)
        elif 'fusion' in name or 'cross_attention' in name:
            fusion_params.append(param)
        else:
            classifier_params.append(param)
    
    # 分层学习率
    optimizer = torch.optim.AdamW([
        {'params': vit_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},  # ViT: 0.1倍
        {'params': temporal_params, 'lr': lr, 'weight_decay': weight_decay},  # 时序: 1倍
        {'params': fusion_params, 'lr': lr, 'weight_decay': weight_decay},  # 融合: 1倍
        {'params': classifier_params, 'lr': lr * 2, 'weight_decay': weight_decay}  # 分类器: 2倍
    ])
    
    return optimizer
```

#### **3.5.2 学习率调度**

```python
def configure_scheduler(optimizer, epochs=50):
    """
    配置学习率调度器：Cosine Annealing + Warmup
    """
    # Warmup
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Cosine Annealing
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    # 组合
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler
```

#### **3.5.3 多卡训练配置**

```python
def setup_distributed_training(model, local_rank):
    """
    配置多卡训练（DDP）
    """
    # 设置设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # 初始化进程组
    torch.distributed.init_process_group(backend='nccl')
    
    # 模型转移到GPU
    model = model.to(device)
    
    # DDP包装
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True  # 因为有些参数可能冻结
    )
    
    return model, device

# 启动命令
# torchrun --nproc_per_node=4 train.py
```

#### **3.5.4 训练循环**

```python
def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_contrast_loss = 0.0
    
    for batch_idx, (images, clinical, labels) in enumerate(train_loader):
        # 数据转移
        images = images.to(device)  # [B, 4, 3, 224, 224]
        clinical = clinical.to(device)  # [B, 22] 原始标准化后的临床特征
        labels = labels.to(device)  # [B]
        
        # 前向传播（返回融合后的特征用于对比学习）
        logits, fused_features = model(images, clinical, return_fused_features=True)
        
        # 分类损失
        cls_loss = criterion(logits, labels)
        
        # 监督对比学习损失（在融合特征上，按MVI标签构造正负样本）
        contrast_loss = model.temporal_transformer.contrastive_loss(
            fused_features, labels  # 注意：输入是融合特征[B,768]，不是时相特征
        )
        
        # 总损失
        loss = cls_loss + 0.1 * contrast_loss  # 对比学习权重0.1
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_contrast_loss += contrast_loss.item()
    
    return {
        'loss': total_loss / len(train_loader),
        'cls_loss': total_cls_loss / len(train_loader),
        'contrast_loss': total_contrast_loss / len(train_loader)
    }
```

---

## 4. 实验设计方案

### 4.1 基线实验

#### **目的**
建立性能基准，验证改进的有效性。

#### **基线模型列表**

| 模型 | 描述 | 预期AUC |
|------|------|---------|
| **B1: 纯临床模型** | Logistic回归（22维特征） | 0.65 |
| **B2: 单时相ResNet18** | 仅使用动脉期图像 | 0.70 |
| **B3: 单时相ViT-B/16** | 仅使用动脉期图像 | 0.72 |
| **B4: 多时相ResNet18（现有）** | 4时相独立处理 + Late Fusion | 0.73 |
| **B5: 多时相ViT-B/16（现有）** | 4时相独立处理 + Late Fusion | **0.76** |
| **B6: MultiModal-ViT（现有）** | ViT + 5维临床特征 | **0.76** |

#### **实验配置**
```python
# 统一配置
BATCH_SIZE = 16
EPOCHS = 50
N_SPLITS = 5  # 5折交叉验证
RANDOM_SEED = 42
OPTIMIZER = AdamW
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
```

---

### 4.2 消融实验

#### **目的**
验证每个创新点的有效性，量化各模块的贡献。

#### **消融实验设计**

| 实验编号 | 模型配置 | 预期AUC | 预期SEN | 预期SPE |
|----------|----------|---------|---------|---------|
| **A0: 完整模型** | 所有创新点 | **0.83** | **0.76** | **0.76** |
| **A1: 无时序Transformer** | 去掉创新点1 | 0.80 | 0.73 | 0.74 |
| **A2: 无Cross-Attention** | 去掉创新点2 | 0.81 | 0.74 | 0.75 |
| **A3: 无改进Focal Loss** | 使用标准CE Loss | 0.82 | 0.70 | 0.78 |
| **A4: 无难样本挖掘** | 去掉Hard Mining | 0.82 | 0.72 | 0.76 |
| **A5: 无对比学习** | 去掉Contrastive Loss | 0.82 | 0.75 | 0.75 |
| **A6: 无特征选择** | 使用全部22维特征 | 0.82 | 0.75 | 0.76 |

#### **消融实验分析**

```python
# 计算每个模块的贡献
贡献度 = AUC(完整模型) - AUC(去掉该模块)

预期结果:
- 时序Transformer: +0.03 (最大贡献)
- Cross-Attention: +0.02
- 改进Focal Loss: +0.01 (主要提升SEN)
- 难样本挖掘: +0.01
- 对比学习: +0.01
- 特征选择: +0.01
```

#### **时相组合实验**

| 实验 | 时相组合 | 预期AUC | 说明 |
|------|----------|---------|------|
| **P1** | 仅grey | 0.68 | 基线 |
| **P2** | 仅ap | 0.72 | 动脉期最重要 |
| **P3** | 仅pp | 0.70 | 门脉期次之 |
| **P4** | 仅lp | 0.69 | 延迟期较弱 |
| **P5** | ap + pp | 0.75 | 两时相组合 |
| **P6** | ap + pp + lp | 0.77 | 三时相组合 |
| **P7** | grey + ap + pp + lp | **0.83** | 全时相（完整模型） |

---

### 4.3 对比实验

#### **目的**
与现有方法对比，证明方法的先进性。

#### **对比方法列表**

| 方法 | 来源 | 描述 | 预期AUC |
|------|------|------|---------|
| **C1: Logistic回归** | 传统方法 | 临床特征 + 手工影像特征 | 0.65 |
| **C2: ResNet18 + 5维临床** | 现有方法 | Late Fusion | 0.73 |
| **C3: ViT-B/16 + 5维临床** | 现有方法（基线） | Late Fusion | **0.76** |
| **C4: ViT-B/16 + 21维临床** | 现有方法 | Late Fusion | 0.77 |
| **C5: 3D CNN** | 文献方法 | 3D卷积建模时序 | 0.78 |
| **C6: LSTM + ViT** | 文献方法 | LSTM建模时序 | 0.79 |
| **C7: 本文方法** | **提出方法** | **时序Transformer + 跨模态融合** | **0.83** |

#### **统计显著性检验**

```python
# 使用DeLong检验比较AUC
from scipy.stats import ttest_rel

# 5折交叉验证的AUC
auc_baseline = [0.75, 0.76, 0.77, 0.76, 0.75]  # 基线
auc_proposed = [0.82, 0.83, 0.84, 0.83, 0.82]  # 提出方法

# t检验
t_stat, p_value = ttest_rel(auc_proposed, auc_baseline)

# 如果p < 0.05，则差异显著
print(f"p-value: {p_value:.4f}")
```

---

### 4.4 可解释性分析

#### **4.4.1 时相权重可视化**

```python
# 可视化不同时相的重要性权重

示例输出:
患者ID: ZS10168702 (MVI+)
时相权重:
- grey: 0.10
- ap:   0.45 ⭐⭐⭐ (最重要)
- pp:   0.30 ⭐⭐
- lp:   0.15 ⭐

解释: 动脉期高增强是MVI+的关键特征
```

**可视化方法**:
- 柱状图：显示每个时相的平均权重
- 热图：显示不同患者的时相权重分布
- 箱线图：对比MVI+和MVI-的时相权重差异

#### **4.4.2 Grad-CAM热图**

```python
# 可视化模型关注的图像区域

步骤:
1. 对每个时相生成Grad-CAM热图
2. 叠加到原始图像上
3. 对比MVI+和MVI-的关注区域

预期发现:
- MVI+: 模型关注肿瘤边缘（血管侵犯区域）
- MVI-: 模型关注肿瘤中心（均匀增强区域）
```

#### **4.4.3 临床特征重要性（SHAP）**

```python
# 使用SHAP值分析临床特征的贡献

import shap

# 计算SHAP值
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data)

# 可视化
shap.summary_plot(shap_values, test_data, feature_names=clinical_feature_names)

预期Top-5特征:
1. 甲胎蛋白 (SHAP: 0.25)
2. 异常凝血酶原 (SHAP: 0.20)
3. HBV (SHAP: 0.15)
4. 年龄 (SHAP: 0.10)
5. 白蛋白 (SHAP: 0.08)
```

#### **4.4.4 跨模态注意力可视化**

```python
# 可视化影像特征对临床特征的注意力

示例输出:
患者ID: ZS10168702 (MVI+)
临床特征注意力权重:
- 甲胎蛋白: 0.35 ⭐⭐⭐
- 异常凝血酶原: 0.28 ⭐⭐⭐
- HBV: 0.15 ⭐⭐
- 年龄: 0.10 ⭐
- 性别: 0.05
- 其他: 0.07

解释: 模型主要关注肿瘤标志物（AFP、PIVKA-II）
```

#### **4.4.5 错误案例分析**

```python
# 分析模型预测错误的案例

错误类型:
1. 假阳性（FP）: 预测MVI+，实际MVI-
   - 可能原因: 临床指标异常但无MVI
   - 典型案例: AFP极高但肿瘤较小

2. 假阴性（FN）: 预测MVI-，实际MVI+
   - 可能原因: 临床指标正常但有MVI
   - 典型案例: AFP正常但影像显示边缘不清

分析方法:
- 统计错误案例的临床特征分布
- 可视化错误案例的Grad-CAM热图
- 分析时相权重是否异常
```

---

### 4.5 评估指标

#### **4.5.1 主要指标**

| 指标 | 公式 | 说明 | 目标值 |
|------|------|------|--------|
| **AUC** | ROC曲线下面积 | 整体排序能力 | **≥0.82** |
| **ACC** | (TP+TN)/(TP+TN+FP+FN) | 整体准确率 | ≥0.78 |
| **SEN** | TP/(TP+FN) | 敏感性（召回率） | **≥0.75** |
| **SPE** | TN/(TN+FP) | 特异性 | **≥0.75** |
| **PPV** | TP/(TP+FP) | 阳性预测值（精确率） | ≥0.70 |
| **NPV** | TN/(TN+FN) | 阴性预测值 | ≥0.85 |
| **F1** | 2×PPV×SEN/(PPV+SEN) | F1分数 | ≥0.72 |

#### **4.5.2 混淆矩阵**

```
预期混淆矩阵（测试集，约40例）:

                实际MVI-    实际MVI+
预测MVI-          24          2      (NPV=0.92)
预测MVI+           6          8      (PPV=0.57)

SEN = 8/(8+2) = 0.80
SPE = 24/(24+6) = 0.80
ACC = (24+8)/40 = 0.80
```

#### **4.5.3 ROC曲线**

```python
# 绘制ROC曲线

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# 绘制
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'本文方法 (AUC = {roc_auc:.3f})')
plt.plot(fpr_baseline, tpr_baseline, label=f'基线方法 (AUC = {auc_baseline:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.xlabel('假阳性率 (1-特异性)')
plt.ylabel('真阳性率 (敏感性)')
plt.title('ROC曲线对比')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png', dpi=300)
```

#### **4.5.4 阈值优化**

```python
# 使用Youden指数优化阈值

from sklearn.metrics import confusion_matrix

def find_optimal_threshold(y_true, y_prob):
    """
    寻找最优阈值（最大化Youden指数）
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    youden_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        youden = sen + spe - 1
        
        youden_scores.append(youden)
    
    optimal_idx = np.argmax(youden_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, youden_scores[optimal_idx]

# 使用
optimal_threshold, max_youden = find_optimal_threshold(y_true, y_prob)
print(f"最优阈值: {optimal_threshold:.3f}")
print(f"最大Youden指数: {max_youden:.3f}")
```

#### **4.5.5 实验记录表格**

```markdown
| 实验 | 模型 | AUC | ACC | SEN | SPE | F1 | 训练时间 | 备注 |
|------|------|-----|-----|-----|-----|----|---------|----|
| B5 | ViT-B/16 (基线) | 0.76 | 0.72 | 0.69 | 0.69 | 0.69 | 50min | 现有最佳 |
| A0 | 完整模型 | 0.83 | 0.78 | 0.76 | 0.76 | 0.76 | 25min | 提出方法 |
| A1 | 无时序Transformer | 0.80 | 0.75 | 0.73 | 0.74 | 0.73 | 20min | 消融实验 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
```

#### **4.5.6 统计分析**

```python
# 5折交叉验证结果的统计分析

import numpy as np

# 5折结果
auc_folds = [0.82, 0.83, 0.84, 0.83, 0.82]
sen_folds = [0.75, 0.76, 0.77, 0.76, 0.75]
spe_folds = [0.75, 0.76, 0.77, 0.76, 0.75]

# 计算均值和标准差
print(f"AUC: {np.mean(auc_folds):.3f} ± {np.std(auc_folds):.3f}")
print(f"SEN: {np.mean(sen_folds):.3f} ± {np.std(sen_folds):.3f}")
print(f"SPE: {np.mean(spe_folds):.3f} ± {np.std(spe_folds):.3f}")

# 95%置信区间
from scipy import stats
confidence = 0.95
ci_auc = stats.t.interval(confidence, len(auc_folds)-1, 
                          loc=np.mean(auc_folds), 
                          scale=stats.sem(auc_folds))
print(f"AUC 95% CI: [{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]")
```

---

## 5. 4个月半月计划

### 5.1 第1阶段（第1-2周）：代码修复 + 真实 Baseline 建立

> ⚠️ **重要调整**: 原计划的"环境搭建+复现"改为"修复关键Bug+建立真实Baseline"。
> 环境已搭建，现有代码已可运行，但存在数据泄露等致命问题（见 1.4 节），
> 必须先修复后才能建立可信的性能基准。

#### **主要任务**
1. 🔴 **修复数据泄露**：KFold 按患者切分（最高优先级）
2. 🔴 **修复临床特征标准化**：所有连续型特征 Z-score
3. 🟡 **修复采样/损失函数配置**：去掉 Sampler，用 class_weights=[1.0, 3.0]
4. 🟢 新建 MultiPhaseDataset 类（为后续时序建模准备）
5. 🟢 跑修正后的 baseline，记录真实性能

#### **详细工作内容**

**Week 1: 代码修复（详见《MVI代码修改方案.md》）**
```bash
任务1.1: 修复数据泄露 (Step 1)
- 修改 vit2.py / resnet18.py / 0318vit2.py / vitResnet.py
- KFold 从图像级别切分 → 患者级别切分
- 添加断言验证无泄露
- 添加患者级别评估（同一患者多张图预测概率取平均）

任务1.2: 修复临床特征处理 (Step 2-3)
- 所有连续型特征 Z-score 标准化
- 缺失值改用中位数填充
- 去掉 WeightedRandomSampler，改用 class_weights=[1.0, 3.0]

任务1.3: 扩大临床 MLP 容量 (Step 4)
- 21→64→128→64 替代 21→32→64

任务1.4: 新增 MultiPhaseDataset (Step 5)
- 按患者×时相组织数据
- 处理缺失时相（零填充 + phase_mask）
- 处理多造影剂（同一时相多张图随机选一）
```

**Week 2: 跑修正后的 Baseline**
```bash
任务2.1: 跑修正后的单图像 baseline
- 使用修正后的 vit2.py（患者级 KFold + 标准化 + class_weights）
- 记录图像级别和患者级别两套指标
- 对比修正前后的差异，验证数据泄露影响程度

任务2.2: 数据探索
- 统计每个患者有多少张图 / 哪些时相 / 哪种造影剂
- 统计时相缺失率
- 分析两种造影剂的图像差异

任务2.3: 实验记录系统
- 创建实验记录表格
- 配置 TensorBoard
- 记录修正后 baseline 作为后续所有实验的对比基准
```

#### **产出物**
- ✅ 修复后的代码（所有4个文件）
- ✅ 修正后 baseline 性能指标（真实值）
- ✅ MultiPhaseDataset 类（备用）
- ✅ 数据分布统计报告（患者数、时相分布、造影剂分布）

#### **关键判断点**
修正后 baseline AUC 预计在 0.65-0.72。根据实际结果决定后续策略：

| 修正后 AUC | 判断 | 策略调整 |
|-----------|------|---------|
| ≥ 0.70 | 现有模型有一定能力 | 按计划推进创新点 |
| 0.60-0.70 | 信号较弱 | 优先等新数据（200例），数据翻倍后再做创新点 |
| < 0.60 | 接近随机 | 需要重新审视数据质量和特征工程 |

#### **产出物**
- ✅ 可运行的基线代码
- ✅ 基线模型权重（AUC=0.76）
- ✅ 数据预处理脚本
- ✅ 实验记录模板

#### **汇报内容**
1. **已完成工作**:
   - 修复数据泄露（按患者分fold）
   - 修复临床特征标准化
   - 修复损失函数配置
   - 建立修正后的 baseline

2. **实验结果**:
   - 修正前 baseline: AUC=0.76（含数据泄露，不可信）
   - 修正后 baseline: AUC=?（真实性能，待填入）
   - 泄露影响程度: AUC 下降了 ?

3. **关键发现**:
   - 数据分布：? 个患者，每个患者平均 ? 张图
   - 时相分布：grey ?%, ap ?%, pp ?%, lp ?%
   - 造影剂分布：sonovue ?%, sonazoid ?%

4. **下一步计划**:
   - 实现多时相时序Transformer
   - 如果修正后 AUC < 0.65，优先等新数据

#### **风险与应对**
- ⚠️ 风险：数据加载速度慢
  - 应对：使用多进程加载（num_workers=4）
- ⚠️ 风险：GPU排队时间长
  - 应对：优先跑关键实验，非关键实验夜间运行

---

### 5.2 第2阶段（第3-4周）：多时相时序Transformer

#### **主要任务**
1. ✅ 实现时序Transformer模块
2. ✅ 实现时序对比学习
3. ✅ 时相权重可视化
4. ✅ 初步实验验证

#### **详细工作内容**

**Week 3: 时序Transformer实现**
```bash
任务3.1: 模块设计
- 实现PositionalEncoding
- 实现TemporalTransformer
- 实现时相权重学习模块

任务3.2: 集成到主模型
- 修改数据加载器（支持多时相输入）
- 修改模型架构（加入时序模块）
- 修改训练循环

任务3.3: 单元测试
- 测试模块输入输出维度
- 测试梯度反向传播
- 测试多卡训练兼容性
```

**Week 4: 对比学习与实验**
```bash
任务4.1: 时序对比学习
- 实现对比学习损失函数
- 调整损失权重（分类损失 vs 对比损失）
- 实验不同温度参数

任务4.2: 实验验证
- 5折交叉验证
- 记录AUC/SEN/SPE
- 对比基线模型

任务4.3: 可视化分析
- 可视化时相权重
- 分析MVI+和MVI-的时相权重差异
- 生成Grad-CAM热图
```

#### **产出物**
- ✅ 时序Transformer代码
- ✅ 训练好的模型权重
- ✅ 实验结果报告
- ✅ 时相权重可视化图

#### **汇报内容**
1. **已完成工作**:
   - 实现多时相时序Transformer
   - 实现时序对比学习
   - 完成5折交叉验证

2. **实验结果**:
   - 模型性能：AUC=0.80±0.02, SEN=0.73±0.03, SPE=0.74±0.02
   - 相比基线提升：AUC +0.04
   - 时相权重：ap(0.45) > pp(0.30) > lp(0.15) > grey(0.10)

3. **关键发现**:
   - 动脉期（ap）对MVI预测最重要
   - MVI+患者的动脉期权重显著高于MVI-

4. **下一步计划**:
   - 实现跨模态动态融合
   - 目标：AUC提升到0.82+

#### **预期指标**
- AUC: 0.76 → **0.80** (+0.04)
- SEN: 0.69 → 0.73 (+0.04)
- SPE: 0.69 → 0.74 (+0.05)

---

### 5.3 第3阶段（第5-6周）：跨模态动态融合

#### **主要任务**
1. ✅ 实现Cross-Attention模块
2. ✅ 实现门控融合机制
3. ✅ 临床特征重要性分析
4. ✅ 实验验证

#### **详细工作内容**

**Week 5: 跨模态融合实现**
```bash
任务5.1: Cross-Attention实现
- 实现CrossModalAttention模块
- 实现注意力权重可视化
- 单元测试

任务5.2: 门控融合实现
- 实现GatedFusion模块
- 实验不同融合策略
- 对比Late Fusion vs Cross-Attention

任务5.3: 集成到主模型
- 修改模型架构
- 修改训练循环
- 测试多卡训练
```

**Week 6: 实验与分析**
```bash
任务6.1: 5折交叉验证
- 训练完整模型
- 记录实验结果
- 对比前一阶段模型

任务6.2: 可解释性分析
- 可视化临床特征注意力权重
- SHAP特征重要性分析
- 分析影像-临床特征交互

任务6.3: 消融实验（初步）
- 无Cross-Attention
- 无门控融合
- 量化各模块贡献
```

#### **产出物**
- ✅ 跨模态融合代码
- ✅ 训练好的模型权重
- ✅ 临床特征重要性分析报告
- ✅ 注意力权重可视化图

#### **汇报内容**
1. **已完成工作**:
   - 实现跨模态Cross-Attention
   - 实现门控融合机制
   - 完成临床特征重要性分析

2. **实验结果**:
   - 模型性能：AUC=0.82±0.02, SEN=0.74±0.03, SPE=0.75±0.02
   - 相比上一阶段提升：AUC +0.02
   - 相比基线提升：AUC +0.06

3. **关键发现**:
   - Top-3重要临床特征：甲胎蛋白(0.35)、异常凝血酶原(0.28)、HBV(0.15)
   - Cross-Attention显著优于Late Fusion

4. **下一步计划**:
   - 实现类别不平衡优化策略
   - 重点提升SEN和SPE

#### **预期指标**
- AUC: 0.80 → **0.82** (+0.02)
- SEN: 0.73 → 0.74 (+0.01)
- SPE: 0.74 → 0.75 (+0.01)

---

### 5.4 第4阶段（第7-8周）：类别不平衡优化 + 第一次汇报

#### **主要任务**
1. ✅ 实现改进的Focal Loss
2. ✅ 实现难样本挖掘
3. ✅ 阈值优化
4. ✅ 准备第一次汇报

#### **详细工作内容**

**Week 7: 类别不平衡优化**
```bash
任务7.1: 改进Focal Loss
- 实现ImprovedFocalLoss
- 实验不同参数（α, γ, λ）
- 对比标准CE Loss

任务7.2: 难样本挖掘
- 实现HardExampleMining
- 实验不同hard_ratio
- 分析难样本特征

任务7.3: 阈值优化
- 实现Youden指数优化
- 绘制SEN-SPE曲线
- 选择最优阈值
```

**Week 8: 实验与汇报准备**
```bash
任务8.1: 完整实验
- 5折交叉验证
- 记录所有指标
- 对比所有基线模型

任务8.2: 汇报材料准备
- 制作PPT（20-30页）
- 整理实验结果表格
- 准备可视化图表
- 准备Demo演示

任务8.3: 第一次汇报
- 汇报已完成工作
- 展示实验结果
- 讨论下一步计划
```

#### **产出物**
- ✅ 类别不平衡优化代码
- ✅ 完整模型权重（前4个月版本）
- ✅ 汇报PPT
- ✅ 实验结果总结报告

#### **汇报内容（重点）**

**1. 项目背景**
- 研究问题：肝癌MVI预测
- 数据规模：201例（MVI+ 51, MVI- 150）
- 技术路线：多模态深度学习

**2. 已完成工作（前8周）**
- ✅ 阶段1：环境搭建 + 基线复现
- ✅ 阶段2：多时相时序Transformer
- ✅ 阶段3：跨模态动态融合
- ✅ 阶段4：类别不平衡优化

**3. 核心创新点**
- 创新点1：多时相时序Transformer（AUC +0.04）
- 创新点2：跨模态动态融合（AUC +0.02）
- 创新点3：类别不平衡优化（SEN/SPE +0.06）

**4. 实验结果**
| 模型 | AUC | SEN | SPE | 提升 |
|------|-----|-----|-----|------|
| 基线（ViT） | 0.76 | 0.69 | 0.69 | - |
| +时序Transformer | 0.80 | 0.73 | 0.74 | +0.04 |
| +跨模态融合 | 0.82 | 0.74 | 0.75 | +0.02 |
| +不平衡优化 | **0.83** | **0.76** | **0.76** | +0.01 |

**5. 可视化展示**
- 时相权重分布图
- 临床特征重要性图
- ROC曲线对比图
- Grad-CAM热图

**6. 下一步计划（后8周）**
- 阶段5：新数据整合（200例）
- 阶段6：消融实验 + 对比实验
- 阶段7：可解释性分析 + 专利撰写
- 阶段8：专利申请 + 论文投稿

**7. 预期最终成果**
- 算法类专利1篇
- 小论文1篇（可选）
- 硕士大论文
- 开源代码（可选）

#### **预期指标**
- AUC: 0.82 → **0.83** (+0.01)
- SEN: 0.74 → **0.76** (+0.02)
- SPE: 0.75 → **0.76** (+0.01)

---

### 5.5 第5阶段（第9-10周）：新数据整合 + 重新训练

#### **主要任务**
1. ✅ 整合新数据（200例）
2. ✅ 数据质量检查
3. ✅ 重新训练模型
4. ✅ 数据增量分析

#### **详细工作内容**

**Week 9: 新数据整合**
```bash
任务9.1: 数据接收与整理
- 接收新数据（约200例）
- 整理多时相图像
- 整理临床数据
- 检查数据完整性

任务9.2: 数据质量检查
- 检查标注质量
- 检查图像质量
- 统计数据分布
- 对比新旧数据差异

任务9.3: 数据预处理
- 统一数据格式
- 合并新旧数据
- 重新划分5折
- 保存预处理数据
```

**Week 10: 重新训练**
```bash
任务10.1: 模型重新训练
- 使用全部数据（~400例）
- 5折交叉验证
- 记录训练日志

任务10.2: 数据增量分析
- 对比新旧数据的模型性能
- 分析数据量对性能的影响
- 绘制学习曲线

任务10.3: 模型优化
- 微调超参数
- 实验不同数据增强策略
- 选择最佳模型
```

#### **产出物**
- ✅ 整合后的完整数据集（~400例）
- ✅ 重新训练的模型权重
- ✅ 数据增量分析报告
- ✅ 学习曲线图

#### **汇报内容**
1. **已完成工作**:
   - 整合新数据200例
   - 数据质量检查完成
   - 模型重新训练完成

2. **实验结果**:
   - 新数据模型性能：AUC=0.84±0.02, SEN=0.77±0.03, SPE=0.77±0.02
   - 相比旧数据提升：AUC +0.01
   - 数据量翻倍带来的提升

3. **关键发现**:
   - 数据量增加显著提升模型稳定性（标准差降低）
   - 新数据分布与旧数据一致

4. **下一步计划**:
   - 完整的消融实验
   - 与SOTA方法对比

#### **预期指标**
- AUC: 0.83 → **0.84** (+0.01)
- SEN: 0.76 → **0.77** (+0.01)
- SPE: 0.76 → **0.77** (+0.01)

---

### 5.6 第6阶段（第11-12周）：消融实验 + 对比实验

#### **主要任务**
1. ✅ 完整的消融实验
2. ✅ 与SOTA方法对比
3. ✅ 统计显著性检验
4. ✅ 实验结果整理

#### **详细工作内容**

**Week 11: 消融实验**
```bash
任务11.1: 模块消融
- A1: 无时序Transformer
- A2: 无Cross-Attention
- A3: 无改进Focal Loss
- A4: 无难样本挖掘
- A5: 无对比学习
- A6: 无特征选择

任务11.2: 时相组合实验
- P1-P7: 不同时相组合
- 分析每个时相的贡献
- 绘制时相贡献图

任务11.3: 结果分析
- 计算每个模块的贡献度
- 统计显著性检验
- 整理消融实验表格
```

**Week 12: 对比实验**
```bash
任务12.1: 基线方法对比
- C1: Logistic回归
- C2: ResNet18 + 临床
- C3: ViT + 临床（基线）
- C4: 3D CNN
- C5: LSTM + ViT

任务12.2: 统计检验
- DeLong检验（AUC）
- t检验（SEN/SPE）
- 计算95%置信区间

任务12.3: 结果可视化
- ROC曲线对比图
- 性能对比柱状图
- 混淆矩阵
```

#### **产出物**
- ✅ 完整的消融实验结果
- ✅ 对比实验结果
- ✅ 统计检验报告
- ✅ 实验结果可视化图

#### **汇报内容**
1. **已完成工作**:
   - 完成6组消融实验
   - 完成5组对比实验
   - 完成统计显著性检验

2. **消融实验结果**:
   - 时序Transformer贡献最大（AUC +0.04）
   - 跨模态融合次之（AUC +0.02）
   - 不平衡优化主要提升SEN/SPE

3. **对比实验结果**:
   - 本文方法显著优于所有基线（p<0.01）
   - 相比最佳基线提升AUC +0.08

4. **下一步计划**:
   - 可解释性深度分析
   - 专利撰写

---

### 5.7 第7阶段（第13-14周）：可解释性分析 + 专利撰写

#### **主要任务**
1. ✅ 深度可解释性分析
2. ✅ 错误案例分析
3. ✅ 专利撰写
4. ✅ 代码整理

#### **详细工作内容**

**Week 13: 可解释性分析**
```bash
任务13.1: Grad-CAM分析
- 生成所有测试样本的热图
- 对比MVI+和MVI-的关注区域
- 分析错误案例的热图

任务13.2: SHAP分析
- 计算所有样本的SHAP值
- 绘制特征重要性图
- 分析特征交互

任务13.3: 注意力权重分析
- 统计时相权重分布
- 统计临床特征注意力分布
- 分析MVI+和MVI-的差异

任务13.4: 错误案例分析
- 统计假阳性和假阴性案例
- 分析错误原因
- 提出改进建议
```

**Week 14: 专利撰写**
```bash
任务14.1: 专利技术交底书
- 撰写技术背景
- 撰写技术方案
- 绘制技术流程图
- 准备附图

任务14.2: 专利申请材料
- 撰写权利要求书
- 撰写说明书
- 撰写说明书摘要
- 准备附图说明

任务14.3: 代码整理
- 整理代码结构
- 添加详细注释
- 编写README
- 准备使用文档
```

#### **产出物**
- ✅ 可解释性分析报告
- ✅ 错误案例分析报告
- ✅ 专利技术交底书
- ✅ 整理后的代码

#### **汇报内容**
1. **已完成工作**:
   - 完成深度可解释性分析
   - 完成错误案例分析
   - 完成专利技术交底书

2. **可解释性发现**:
   - 模型主要关注肿瘤边缘区域
   - 甲胎蛋白和异常凝血酶原最重要
   - 动脉期对MVI+预测最关键

3. **错误案例分析**:
   - 假阳性：临床指标异常但无MVI（6例）
   - 假阴性：临床指标正常但有MVI（2例）

4. **下一步计划**:
   - 提交专利申请
   - 准备论文投稿（可选）

---

### 5.8 第8阶段（第15-16周）：专利申请 + 论文投稿准备

#### **主要任务**
1. ✅ 提交专利申请
2. ✅ 论文撰写（可选）
3. ✅ 最终汇报准备
4. ✅ 项目总结

#### **详细工作内容**

**Week 15: 专利申请**
```bash
任务15.1: 专利申请
- 提交专利申请材料
- 配合专利代理人修改
- 完成专利申请流程

任务15.2: 论文撰写（可选）
- 撰写论文初稿
- 准备论文图表
- 选择投稿期刊
- 准备投稿材料

任务15.3: 代码开源准备（可选）
- 代码最终整理
- 编写详细文档
- 准备预训练模型
- 准备示例数据
```

**Week 16: 项目总结**
```bash
任务16.1: 最终汇报准备
- 制作汇报PPT（40-50页）
- 整理所有实验结果
- 准备Demo演示
- 准备答辩材料

任务16.2: 项目总结
- 撰写项目总结报告
- 整理所有产出物
- 归档实验数据
- 备份代码和模型

任务16.3: 最终汇报
- 汇报4个月工作
- 展示最终成果
- 讨论后续工作
```

#### **产出物**
- ✅ 专利申请受理通知书
- ✅ 论文初稿（可选）
- ✅ 最终汇报PPT
- ✅ 项目总结报告

#### **最终汇报内容**

**1. 项目概述**
- 研究问题：肝癌MVI预测
- 数据规模：~400例
- 时间跨度：4个月（16周）

**2. 核心创新点**
- 创新点1：多时相时序Transformer
- 创新点2：跨模态动态融合
- 创新点3：类别不平衡优化策略
- 辅助创新点4-5

**3. 最终实验结果**
| 指标 | 基线 | 最终 | 提升 |
|------|------|------|------|
| AUC | 0.76 | **0.84** | +0.08 |
| ACC | 0.72 | **0.80** | +0.08 |
| SEN | 0.69 | **0.77** | +0.08 |
| SPE | 0.69 | **0.77** | +0.08 |

**4. 主要产出**
- ✅ 算法类专利1篇（已申请）
- ✅ 完整的实验代码
- ✅ 训练好的模型权重
- ✅ 详细的实验报告
- ✅ 可解释性分析报告

**5. 后续工作（大论文）**
- 补充文献综述
- 补充理论分析
- 补充实验细节
- 撰写大论文（8-10万字）

**6. 时间规���（后2个月）**
- 月1：补充实验 + 大论文撰写
- 月2：大论文修改 + 答辩准备

---

## 6. 论文与专利策略

### 6.1 算法类专利

#### **专利名称**
```
"一种基于多时相超声造影和跨模态注意力的肝癌微血管侵犯预测方法"
```

#### **专利类型**
- 发明专利（算法类）

#### **核心技术点**

**技术点1：多时相时序Transformer**
- 时序自注意力机制
- 自适应时相权重学习
- 时序对比学习

**技术点2：跨模态动态融合**
- Cross-Attention机制
- 门控融合策略
- 特征重要性可视化

**技术点3：类别不平衡优化**
- 改进的Focal Loss（多模态版）
- 难样本挖掘策略
- 阈值优化方法

#### **权利要求书（初稿）**

**独立权利要求1**：
```
一种基于多时相超声造影和跨模态注意力的肝癌微血管侵犯预测方法，
其特征在于，包括以下步骤：

步骤1：获取患者的多时相超声造影图像和临床特征数据；
步骤2：使用预训练的视觉Transformer提取每个时相的影像特征；
步骤3：使用时序Transformer对多时相特征进行时序建模，学习增强-消退模式；
步骤4：使用跨模态注意力机制融合影像特征和临床特征；
步骤5：使用改进的Focal Loss训练分类器，输出MVI预测结果。
```

**从属权利要求2-10**：
- 权利要求2：时序Transformer的具体实现
- 权利要求3：时相权重学习方法
- 权利要求4：时序对比学习方法
- 权利要求5：跨模态注意力机制
- 权利要求6：门控融合策略
- 权利要求7：改进Focal Loss的计算方法
- 权利要求8：难样本挖掘策略
- 权利要求9：阈值优化方法
- 权利要求10：不确定性量化方法

#### **技术流程图**

```
┌─────────────────────────────────────────────────────────────┐
│                    输入数据                                  │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │ 多时相CEUS图像   │        │ 临床特征数据     │          │
│  │ [grey,ap,pp,lp]  │        │ (22维)           │          │
│  └──────────────────┘        └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                ↓                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  特征提取                                    │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │ ViT-B/16         │        │ 临床MLP          │          │
│  │ [B,4,768]        │        │ [B,64]           │          │
│  └──────────────────┘        └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│          技术点1: 多时相时序Transformer                      │
│  ┌──────────────────────────────────────────────┐          │
│  │ 时序编码 → 自注意力 → 时相权重学习          │          │
│  │ [B,4,768] → [B,768]                          │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                ↓                         ↓
┌─────────────────────────────────────────────────────────────┐
│          技术点2: 跨模态动态融合                             │
│  ┌──────────────────────────────────────────────┐          │
│  │ Cross-Attention + 门控融合                   │          │
│  │ 影像[B,768] × 临床[B,64] → [B,832]           │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│                  分类器                                      │
│  ┌──────────────────────────────────────────────┐          │
│  │ MLP: 832 → 256 → 128 → 2                     │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│          技术点3: 类别不平衡优化                             │
│  ┌──────────────────────────────────────────────┐          │
│  │ 改进Focal Loss + 难样本挖掘 + 阈值优化      │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│                  输出结果                                    │
│  ┌──────────────────────────────────────────────┐          │
│  │ MVI预测: MVI+ / MVI-                         │          │
│  │ 预测概率: [0, 1]                             │          │
│  │ 置信度: [0, 1]                               │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

#### **申请时间表**

| 阶段 | 时间 | 任务 |
|------|------|------|
| 技术交底书 | 第13-14周 | 撰写技术方案、绘制流程图 |
| 专利申请材料 | 第14周 | 撰写权利要求书、说明书 |
| 提交申请 | 第15周 | 提交专利局 |
| 受理通知 | 第16周 | 获得受理通知书 |
| 实质审查 | 6-12个月 | 配合审查意见 |
| 授权 | 12-24个月 | 获得专利证书 |

#### **预期成果**
- ✅ 专利申请号
- ✅ 专利受理通知书
- ✅ 作为硕士大论文的创新点证明

---

### 6.2 小论文（可选）

#### **论文标题**
```
"Multi-Phase CEUS with Temporal Transformer and Cross-Modal 
Attention for Microvascular Invasion Prediction in Hepatocellular Carcinoma"

（基于时序Transformer和跨模态注意力的多时相超声造影肝癌微血管侵��预测）
```

#### **推荐投稿期刊**

**首选：IEEE Access**
- 类型：SCI Q2期刊
- 影响因子：~3.5
- 审稿周期：2-3个月
- 接受率：~40%
- 优势：开源期刊、审稿快、容易中

**备选1：Biomedical Signal Processing and Control**
- 类型：SCI Q2期刊
- 影响因子：~4.5
- 审稿周期：3-4个月
- 优势：专注于生物医学信号处理

**备选2：Computers in Biology and Medicine**
- 类型：SCI Q2期刊
- 影响因子：~7.0
- 审稿周期：4-6个月
- 优势：影响因子较高

**备选3：会议论文**
- ICASSP 2026（截稿：2025年10月）
- ICIP 2026（截稿：2025年11月）
- ISBI 2026（截稿：2025年10月）

#### **论文结构**

**Abstract（200-250词）**
- 背景：MVI预测的临床重要性
- 问题：现有方法的局限性
- 方法：多时相时序Transformer + 跨模态融合
- 结果：AUC=0.84, SEN=0.77, SPE=0.77
- 结论：显著优于现有方法

**1. Introduction**
- 1.1 研究背景
- 1.2 相关工作
- 1.3 研究动机
- 1.4 主要贡献

**2. Materials and Methods**
- 2.1 数据集
- 2.2 多时相时序Transformer
- 2.3 跨模态动态融合
- 2.4 类别不平衡优化
- 2.5 训练策略

**3. Experiments**
- 3.1 实验设置
- 3.2 评估指标
- 3.3 基线方法对比
- 3.4 消融实验
- 3.5 可解释性分析

**4. Results**
- 4.1 整体性能
- 4.2 消融实验结果
- 4.3 对比实验结果
- 4.4 可视化分析

**5. Discussion**
- 5.1 主要发现
- 5.2 临床意义
- 5.3 局限性
- 5.4 未来工作

**6. Conclusion**

#### **投稿时间表**

| 阶段 | 时间 | 任务 |
|------|------|------|
| 论文撰写 | 第14-15周 | 撰写初稿 |
| 内部审阅 | 第15周 | 导师审阅、修改 |
| 投稿 | 第16周 | 提交期刊 |
| 审稿 | 2-4个月 | 等待审稿意见 |
| 修改 | 1-2周 | 根据审稿意见修改 |
| 接收 | 3-6个月 | 论文接收 |

---

### 6.3 硕士大论文

#### **论文标题**
```
"基于多模态深度学习的肝细胞癌微血管侵犯预测研究"
```

#### **论文结构（8-10万字）**

**第1章：绪论（1.5万字）**
- 1.1 研究背景与意义
  - 肝细胞癌的流行病学
  - 微血管侵犯的临床重要性
  - 术前预测的必要性
- 1.2 国内外研究现状
  - 传统影像学方法
  - 机器学习方法
  - 深度学习方法
  - 多模态融合方法
- 1.3 研究内容与目标
- 1.4 论文组织结构

**第2章：相关理论与技术（2万字）**
- 2.1 超声造影成像原理
  - 超声造影剂
  - 多时相成像
  - 增强模式分析
- 2.2 深度学习基础
  - 卷积神经网络
  - Transformer架构
  - 注意力机制
- 2.3 多模态学习
  - 特征融合策略
  - 跨模态学习
  - 多模态预训练
- 2.4 类别不平衡处理
  - 重采样方法
  - 损失函数设计
  - 难样本挖掘
- 2.5 模型可解释性
  - Grad-CAM
  - SHAP
  - 注意力可视化

**第3章：数据采集与预处理（1.5万字）**
- 3.1 数据采集
  - 患者纳入/排除标准
  - 影像采集方案
  - 临床数据收集
  - 病理金标准
- 3.2 数据统计分析
  - 基线特征对比
  - 单因素分析
  - 相关性分析
- 3.3 数据预处理
  - 图像预处理
  - 临床特征处理
  - 数据增强
  - 数据划分

**第4章：多时相时序建模方法（2万字）**
- 4.1 问题分析
  - 多时相数据的特点
  - 时序建模的必要性
- 4.2 时序Transformer设计
  - 时序编码
  - 自注意力机制
  - 时相权重学习
- 4.3 时序对比学习
  - 对比学习原理
  - 正负样本构造
  - 损失函数设计
- 4.4 实验验证
  - 实验设置
  - 结果分析
  - 消融实验

**第5章：跨模态融合方法（1.5万字）**
- 5.1 问题分析
  - 多模态融合的挑战
  - 现有方法的局限性
- 5.2 跨模态注意力设计
  - Cross-Attention机制
  - 门控融合策略
  - 特征重要性学习
- 5.3 实验验证
  - 实验设置
  - 结果分析
  - 可视化分析

**第6章：类别不平衡优化方法（1万字）**
- 6.1 问题分析
  - 样本不平衡的影响
  - 现有方法的问题
- 6.2 改进Focal Loss设计
  - 多模态版Focal Loss
  - 临床置信度调制
- 6.3 难样本挖掘策略
  - 难样本定义
  - 挖掘策略设计
- 6.4 阈值优化方法
  - Youden指数
  - 最优阈值选择
- 6.5 实验验证

**第7章：完整系统实现与实验（2万字）**
- 7.1 系统架构设计
  - 整体架构
  - 模块设计
  - 训练策略
- 7.2 实验设置
  - 数据集
  - 评估指标
  - 实现细节
- 7.3 基线方法对比
  - 传统方法
  - 深度学习方法
  - 统计显著性检验
- 7.4 消融实验
  - 各模块贡献分析
  - 时相组合实验
  - 特征选择实验
- 7.5 可解释性分析
  - Grad-CAM分析
  - SHAP分析
  - 注意力权重分析
  - 错误案例分析
- 7.6 结果讨论
  - 主要发现
  - 临床意义
  - 局限性分析

**第8章：总结与展望（0.5万字）**
- 8.1 研究工作总结
- 8.2 主要创新点
- 8.3 研究局限性
- 8.4 未来工作展望

**参考文献**
- 预计100-150篇

**附录**
- 附录A：数据统计表
- 附录B：实验结果详表
- 附录C：代码清单
- 附录D：发表论文/专利

#### **撰写时间表（后2个月）**

| 周次 | 章节 | 任务 |
|------|------|------|
| 第17-18周 | 第1-2章 | 绪论 + 相关理论 |
| 第19-20周 | 第3-4章 | 数据 + 时序建模 |
| 第21-22周 | 第5-6章 | 跨模态融合 + 不平衡优化 |
| 第23-24周 | 第7-8章 | 完整实验 + 总结 |
| 第25周 | 全文 | 修改润色 |
| 第26周 | 全文 | 导师审阅、最终修改 |

#### **预期成果**
- ✅ 硕士学位论文（8-10万字）
- ✅ 满足毕业要求
- ✅ 包含专利和小论文（加分项）

---

## 7. 风险评估与应对

### 7.1 技术风险

#### **风险1：模型性能未达预期目标** ⚠️⚠️⚠️
**风险描述**：
- 目标AUC=0.82+，但实际可能只达到0.78-0.80
- SEN/SPE未能同时达到0.75+

**可能原因**：
1. 数据量不足（~400例相对较少）
2. 样本不平衡问题未完全解决
3. 模型设计不够优化

**应对策略**：
- ✅ **Plan A**：调整目标（AUC=0.80也是显著提升）
- ✅ **Plan B**：增加数据增强策略
- ✅ **Plan C**：尝试集成学习（多模型融合）
- ✅ **Plan D**：调整评估指标（如F1-score、Youden指数）

**风险等级**：中等  
**影响**：不影响毕业，但可能影响论文质量

---

#### **风险2：多卡训练资源竞争** ⚠️⚠️
**风险描述**：
- 8张A800需要排队使用
- 关键实验可能无法及时完成

**可能原因**：
1. 实验室其他同学同时使用GPU
2. 训练时间过长占用资源
3. GPU故障或维护

**应对策略**：
- ✅ **优先级管理**：关键实验优先，非关键实验夜间运行
- ✅ **时间规划**：提前预约GPU使用时间
- ✅ **代码优化**：减少训练时间（混合精度、梯度累积）
- ✅ **备用方案**：单卡训练（时间更长但可行）

**风险等级**：中等  
**影响**：可能延长实验时间1-2周

---

#### **风险3：新数据质量问题** ⚠️⚠️
**风险描述**：
- 新增200例数据可能存在标注错误
- 数据分布与现有数据不一致

**可能原因**：
1. 不同医生标注标准不一致
2. 图像采集设备或参数不同
3. 患者群体差异

**应对策略**：
- ✅ **数据检查**：仔细检查新数据质量
- ✅ **分布对比**：对比新旧数据的统计分布
- ✅ **分阶段实验**：先用旧数据验证方法，再加入新数据
- ✅ **数据清洗**：剔除明显错误的数据

**风险等级**：低  
**影响**：可能需要1周时间清洗数据

---

#### **风险4：代码实现bug** ⚠️
**风险描述**：
- 模块实现存在bug导致结果不可靠
- 多卡训练同步问题

**可能原因**：
1. 代码逻辑错误
2. 维度不匹配
3. 梯度消失/爆炸

**应对策略**：
- ✅ **单元测试**：每个模块编写测试代码
- ✅ **维度检查**：打印中间层输出维度
- ✅ **梯度检查**：监控梯度范数
- ✅ **代码审查**：请导师或同学帮忙审查

**风险等级**：低  
**影响**：可能需要1-2天调试

---

### 7.2 时间风险

#### **风险5：实验进度延迟** ⚠️⚠️⚠️
**风险描述**：
- 某个阶段的实验未能按时完成
- 影响后续阶段的进度

**可能原因**：
1. 技术难度超出预期
2. GPU资源不足
3. 个人时间安排冲突

**应对策略**：
- ✅ **缓冲时间**：每个阶段预留2-3天缓冲
- ✅ **并行工作**：部分任务可以并行进行
- ✅ **优先级调整**：关键任务优先，非关键任务可延后
- ✅ **简化方案**：必要时简化部分创新点

**时间缓冲计划**：
| 阶段 | 计划时间 | 缓冲时间 | 最晚完成时间 |
|------|----------|----------|--------------|
| 阶段1-2 | 4周 | +3天 | 4.5周 |
| 阶段3-4 | 4周 | +3天 | 4.5周 |
| 阶段5-6 | 4周 | +3天 | 4.5周 |
| 阶段7-8 | 4周 | +3天 | 4.5周 |

**风险等级**：高  
**影响**：可能延长项目时间1-2周

---

#### **风险6：专利申请流程延误** ⚠️
**风险描述**：
- 专利申请材料准备时间不足
- 专利代理人沟通延误

**可能原因**：
1. 技术交底书撰写困难
2. 专利代理人修改意见多
3. 学校审批流程慢

**应对策略**：
- ✅ **提前准备**：第13周开始准备技术交底书
- ✅ **模板参考**：参考类似专利的格式
- ✅ **及时沟通**：与专利代理人保持沟通
- ✅ **备用方案**：如果4个月内无法完成，可延后到大论文阶段

**风险等级**：低  
**影响**：不影响毕业，但可能影响专利申请时间

---

### 7.3 资源风险

#### **风险7：计算资源不足** ⚠️⚠️
**风险描述**：
- GPU排队时间过长
- 存储空间不足

**可能原因**：
1. 实验室GPU使用高峰期
2. 模型权重和日志占用大量存储
3. 数据集过大

**应对策略**：
- ✅ **错峰使用**：夜间或周末运行实验
- ✅ **存储管理**：定期清理无用文件
- ✅ **云端备份**：重要数据备份到云端
- ✅ **代码优化**：减少显存占用

**存储空间估算**：
| 项目 | 大小 | 说明 |
|------|------|------|
| 原始数据 | ~10GB | 图像 + 临床数据 |
| 预处理数据 | ~15GB | 增强后的数据 |
| 模型权重 | ~5GB | 多个模型版本 |
| 实验日志 | ~2GB | TensorBoard日志 |
| 总计 | ~32GB | 需要至少50GB空间 |

**风险等级**：低  
**影响**：可能需要清理存储空间

---

#### **风险8：临床医生配合度** ⚠️
**风险描述**：
- 新数据延迟提供
- 标注质量不符合要求

**可能原因**：
1. 临床医生工作繁忙
2. 数据收集困难
3. 标注标准不明确

**应对策略**：
- ✅ **提前沟通**：提前与临床医生沟通数据需求
- ✅ **明确标准**：提供清晰的标注标准
- ✅ **定期跟进**：定期询问数据收集进度
- ✅ **备用方案**：如果新数据延迟，先用现有数据完成实验

**风险等级**：中等  
**影响**：可能延迟1-2周

---

#### **风险9：个人健康或突发事件** ⚠️
**风险描述**：
- 生病或其他突发事件影响进度

**应对策略**：
- ✅ **健康管理**：保持良好作息，避免过度劳累
- ✅ **时间缓冲**：预留缓冲时间应对突发情况
- ✅ **及时沟通**：如有突发情况及时与导师沟通

**风险等级**：低  
**影响**：视具体情况而定

---

### 7.4 风险总结与应对矩阵

| 风险 | 等级 | 概率 | 影响 | 应对策略 | 责任人 |
|------|------|------|------|----------|--------|
| 模型性能未达预期 | 中 | 30% | 中 | 调整目标/增强数据 | 自己 |
| GPU资源竞争 | 中 | 50% | 中 | 错峰使用/优先级管理 | 自己 |
| 新数据质量问题 | 低 | 20% | 低 | 数据检查/清洗 | 自己+医生 |
| 代码bug | 低 | 30% | 低 | 单元测试/代码审查 | 自己 |
| 实验进度延迟 | 高 | 40% | 高 | 缓冲时间/并行工作 | 自己 |
| 专利申请延误 | 低 | 20% | 低 | 提前准备/及时沟通 | 自己+代理人 |
| 计算资源不足 | 低 | 20% | 低 | 存储管理/代码优化 | 自己 |
| 临床医生配合 | 中 | 30% | 中 | 提前沟通/定期跟进 | 医生 |
| 个人突发事件 | 低 | 10% | 高 | 时间缓冲/及时沟通 | 自己 |

**总体风险评估**：
- 高风险：1个（实验进度延迟）
- 中风险：3个（模型性能、GPU资源、临床配合）
- 低风险：5个

**风险应对原则**：
1. ✅ **预防为主**：提前规划，避免风险发生
2. ✅ **及时应对**：发现风险及时调整策略
3. ✅ **灵活调整**：根据实际情况调整计划
4. ✅ **保持沟通**：与导师和临床医生保持沟通

---

## 8. 附录：代码实现指南

### 8.1 环境配置

#### **Python环境**
```bash
# 创建conda环境
conda create -n mvi python=3.9
conda activate mvi

# 安装PyTorch（CUDA 11.7）
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 安装依赖包
pip install timm==0.6.12  # 预训练模型
pip install transformers==4.25.1  # Transformer模型
pip install tensorboard==2.11.0  # 可视化
pip install scikit-learn==1.2.0  # 机器学习工具
pip install pandas==1.5.2  # 数据处理
pip install numpy==1.23.5  # 数值计算
pip install matplotlib==3.6.2  # 绘图
pip install seaborn==0.12.1  # 统计绘图
pip install shap==0.41.0  # 可解释性分析
pip install opencv-python==4.7.0.68  # 图像处理
pip install pillow==9.3.0  # 图像读取
pip install openpyxl==3.0.10  # Excel读取
pip install tqdm==4.64.1  # 进度条
```

#### **验证安装**
```python
import torch
import torchvision
import timm
import transformers

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"可用GPU数量: {torch.cuda.device_count()}")
print(f"GPU型号: {torch.cuda.get_device_name(0)}")
```

---

### 8.2 多卡训练配置

#### **启动多卡训练**
```bash
# 使用torchrun启动（推荐）
torchrun --nproc_per_node=4 train.py --config config.yaml

# 或使用python -m torch.distributed.launch（旧版）
python -m torch.distributed.launch --nproc_per_node=4 train.py --config config.yaml
```

#### **训练脚本示例**
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """初始化分布式训练"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """清理分布式训练"""
    dist.destroy_process_group()

def main():
    # 初始化
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # 创建模型
    model = YourModel().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 创建数据加载器（使用DistributedSampler）
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    
    # 训练循环
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # 重要！确保每个epoch的数据打乱
        train_one_epoch(model, train_loader, optimizer, device)
    
    # 清理
    cleanup_distributed()

if __name__ == '__main__':
    main()
```

#### **注意事项**
1. ✅ 使用`DistributedSampler`确保数据不重复
2. ✅ 每个epoch调用`sampler.set_epoch(epoch)`
3. ✅ 只在rank=0的进程保存模型和日志
4. ✅ 使用`dist.barrier()`同步进程

---

### 8.3 TensorBoard使用

#### **启动TensorBoard**
```bash
# 启动TensorBoard服务
tensorboard --logdir=runs --port=6006

# 在浏览器中访问
# http://localhost:6006
```

#### **记录训练日志**
```python
from torch.utils.tensorboard import SummaryWriter

# 创建writer
writer = SummaryWriter('runs/experiment_name')

# 记录标量
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('AUC/train', auc, epoch)
writer.add_scalar('SEN/train', sen, epoch)
writer.add_scalar('SPE/train', spe, epoch)

# 记录学习率
writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

# 记录图像
writer.add_image('GradCAM', gradcam_image, epoch)

# 记录直方图
writer.add_histogram('Attention_Weights', attention_weights, epoch)

# 记录模型图
writer.add_graph(model, input_tensor)

# 关闭writer
writer.close()
```

#### **可视化示例**
```python
# 绘制ROC曲线
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png', dpi=300)
plt.close()

# 添加到TensorBoard
writer.add_figure('ROC_Curve', plt.gcf(), epoch)
```

---

### 8.4 代码结构建议

#### **推荐的项目结构**
```
MVI/
├── data/                          # 数据目录
│   ├── images/                    # 图像数据
│   │   ├── MVI+/
│   │   └── MVI-/
│   ├── clinical/                  # 临床数据
│   │   ├── MVI1.xlsx
│   │   └── MVI0.xlsx
│   └── processed/                 # 预处理后的数据
│
├── models/                        # 模型定义
│   ├── __init__.py
│   ├── temporal_transformer.py   # 时序Transformer
│   ├── cross_modal_fusion.py     # 跨模态融合
│   ├── focal_loss.py              # 改进Focal Loss
│   └── full_model.py              # 完整模型
│
├── utils/                         # 工具函数
│   ├── __init__.py
│   ├── dataset.py                 # 数据加载器
│   ├── metrics.py                 # 评估指标
│   ├── visualization.py           # 可视化工具
│   └── logger.py                  # 日志工具
│
├── configs/                       # 配置文件
│   ├── config.yaml                # 主配置
│   └── config_baseline.yaml       # 基线配置
│
├── scripts/                       # 脚本
│   ├── train.py                   # 训练脚本
│   ├── test.py                    # 测试脚本
│   ├── evaluate.py                # 评估脚本
│   └── visualize.py               # 可视化脚本
│
├── notebooks/                     # Jupyter notebooks
│   ├── data_analysis.ipynb        # 数据分析
│   ├── model_analysis.ipynb       # 模型分析
│   └── results_visualization.ipynb # 结果可视化
│
├── checkpoints/                   # 模型权重
│   ├── baseline/
│   ├── temporal/
│   └── full_model/
│
├── runs/                          # TensorBoard日志
│   ├── experiment_1/
│   ├── experiment_2/
│   └── ...
│
├── results/                       # 实验结果
│   ├── figures/                   # 图表
│   ├── tables/                    # 表格
│   └── reports/                   # 报告
│
├── requirements.txt               # 依赖包
├── README.md                      # 项目说明
└── train.sh                       # 训练脚本
```

#### **配置文件示例（config.yaml）**
```yaml
# 数据配置
data:
  image_dir: "data/images"
  clinical_file: "data/clinical/MVI1.xlsx"
  image_size: 224
  batch_size: 16
  num_workers: 4
  
# 模型配置
model:
  backbone: "vit_base_patch16_224"
  pretrained: true
  num_phases: 4
  clinical_dim: 22
  hidden_dim: 768
  num_heads: 8
  num_layers: 2
  dropout: 0.1
  
# 训练配置
training:
  epochs: 50
  learning_rate: 1e-4
  weight_decay: 0.05
  warmup_epochs: 5
  n_splits: 5
  random_seed: 42
  
# 损失函数配置
loss:
  type: "improved_focal"
  alpha: 3.0
  gamma: 2.0
  lambda_clinical: 0.5
  contrast_weight: 0.1
  
# 优化器配置
optimizer:
  type: "adamw"
  lr_vit: 1e-5
  lr_temporal: 1e-4
  lr_fusion: 1e-4
  lr_classifier: 2e-4
  
# 日志配置
logging:
  log_dir: "runs"
  save_dir: "checkpoints"
  log_interval: 10
  save_interval: 5
```

#### **训练脚本示例（train.py）**
```python
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 解析参数
    args = parse_args()
    config = load_config(args.config)
    
    # 设置设备
    if args.local_rank != -1:
        # 多卡训练
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        torch.distributed.init_process_group(backend='nccl')
    else:
        # 单卡训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    train_dataset = YourDataset(config, split='train')
    val_dataset = YourDataset(config, split='val')
    
    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], 
                                 sampler=train_sampler, num_workers=config['data']['num_workers'])
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], 
                                 shuffle=True, num_workers=config['data']['num_workers'])
    
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], 
                           shuffle=False, num_workers=config['data']['num_workers'])
    
    # 创建模型
    model = YourModel(config).to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    
    # 创建优化器和调度器
    optimizer = configure_optimizer(model, config)
    scheduler = configure_scheduler(optimizer, config)
    
    # 创建损失函数
    criterion = ImprovedFocalLoss(config)
    
    # 创建TensorBoard writer
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(config['logging']['log_dir'])
    
    # 训练循环
    best_auc = 0.0
    for epoch in range(config['training']['epochs']):
        # 训练
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # 验证
        val_metrics = validate(model, val_loader, device, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        if args.local_rank in [-1, 0]:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
            writer.add_scalar('SEN/val', val_metrics['sen'], epoch)
            writer.add_scalar('SPE/val', val_metrics['spe'], epoch)
            
            # 保存最佳模型
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                torch.save(model.state_dict(), f"{config['logging']['save_dir']}/best_model.pth")
                print(f"Saved best model with AUC={best_auc:.4f}")
    
    # 清理
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()
    if args.local_rank in [-1, 0]:
        writer.close()

if __name__ == '__main__':
    main()
```

#### **使用说明**
```bash
# 1. 数据预处理
python scripts/preprocess.py --config configs/config.yaml

# 2. 训练模型（单卡）
python scripts/train.py --config configs/config.yaml

# 3. 训练模型（多卡）
torchrun --nproc_per_node=4 scripts/train.py --config configs/config.yaml

# 4. 测试模型
python scripts/test.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth

# 5. 可视化结果
python scripts/visualize.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth

# 6. 启动TensorBoard
tensorboard --logdir=runs --port=6006
```

---

## 9. 总结

### 9.1 项目概述

本文档详细规划了一个为期4个月的肝癌微血管侵犯（MVI）预测项目，旨在完成硕士毕业大论文和算法类专利申请。

**核心目标**：
- ✅ 提升模型性能：AUC从0.76提升到0.84+
- ✅ 保证创新点：5个创新点（3主+2辅）
- ✅ 保证工作量：足够支撑硕士大论文
- ✅ 完成专利申请：算法类发明专利

### 9.2 核心创新点

1. **多时相时序Transformer**：建模时序演变模式，自动学习时相权重
2. **跨模态动态融合**：Cross-Attention + 门控融合，深度交互影像和临床特征
3. **类别不平衡优化**：改进Focal Loss + 难样本挖掘 + 阈值优化
4. **连续值特征智能处理**：特征选择 + 自适应加权 + 特征交互
5. **不确定性量化**：Monte Carlo Dropout + 集成学习 + 置信度输出

### 9.3 时间规划

| 阶段 | 时间 | 主要任务 | 产出 |
|------|------|----------|------|
| 阶段1 | 第1-2周 | 环境搭建 + 基线复现 | 基线代码 + 模型权重 |
| 阶段2 | 第3-4周 | 多时相时序Transformer | AUC提升到0.80 |
| 阶段3 | 第5-6周 | 跨模态动态融合 | AUC提升到0.82 |
| 阶段4 | 第7-8周 | 类别不平衡优化 + 第一次汇报 | AUC提升到0.83 |
| 阶段5 | 第9-10周 | 新数据整合 + 重新训练 | AUC提升到0.84 |
| 阶段6 | 第11-12周 | 消融实验 + 对比实验 | 完整实验结果 |
| 阶段7 | 第13-14周 | 可解释性分析 + 专利撰写 | 专利技术交底书 |
| 阶段8 | 第15-16周 | 专利申请 + 论文投稿准备 | 专利受理通知书 |

### 9.4 预期成果

**学术成果**：
- ✅ 算法类发明专利1篇（已申请）
- ✅ 小论文1篇（可选，IEEE Access或会议）
- ✅ 硕士学位论文1篇（8-10万字）

**技术成果**：
- ✅ 完整的实验代码
- ✅ 训练好的模型权重
- ✅ 详细的实验报告
- ✅ 可解释性分析报告

**性能指标**：
| 指标 | 基线 | 目标 | 实际预期 |
|------|------|------|----------|
| AUC | 0.76 | 0.82+ | **0.84** |
| ACC | 0.72 | 0.78+ | **0.80** |
| SEN | 0.69 | 0.75+ | **0.77** |
| SPE | 0.69 | 0.75+ | **0.77** |

### 9.5 关键成功因素

1. ✅ **技术路线清晰**：5个创新点层层递进
2. ✅ **时间规划合理**：每个阶段2周，预留缓冲时间
3. ✅ **风险管理到位**：识别9个主要风险，制定应对策略
4. ✅ **资源保障充足**：8张A800 GPU，临床医生支持
5. ✅ **目标切实可行**：不追求顶刊，但保证创新性

### 9.6 后续工作（大论文阶段）

**时间**：第17-26周（后2个月）

**主要任务**：
1. 补充文献综述
2. 补充理论分析
3. 补充实验细节
4. 撰写大论文（8-10万字）
5. 论文修改和答辩准备

---

**文档状态**: 🔄 v1.1 — 代码审查后修正版

**文档版本**: v1.1

**创建时间**: 2026-05-03

**最后更新**: 2026-05-03 (v1.1: 修正数据泄露、CrossAttention设计、对比学习策略、参数量估算)

**v1.1 主要变更**:
- 新增 §1.4: 代码审查发现的关键问题（数据泄露、特征标准化、采样矛盾）
- 修正 §2.2.3: 对比学习改为监督对比学习（SupCon），避免破坏时相差异信号
- 修正 §2.3.1: CrossAttention 改为逐指标 token 设计，修复 softmax 退化问题
- 修正 §3.1: ViT 参数量改为权重共享（89M 而非 344M）
- 修正 §3.2: contrastive_loss 实现与设计对齐
- 修正 §3.3: CrossModalAttention 实现与设计对齐
- 更新 §1.5: 性能目标改为区间（待修正后 baseline 确定）
- 更新 §5.1: 第一阶段改为代码修复 + 真实 baseline 建立
- 配套文档: 新增《MVI代码修改方案.md》（给 Claude Code 的完整指令）

---

**联系方式**：
- 学生：[你的姓名]
- 导师：[导师姓名]
- 合作医生：[医生姓名]

**项目地址**：
- 代码仓库：[待定]
- 实验记录：[待定]
- 文档地址：/home/fuxiangyu/jlx/MVI/MVI项目技术方案与4个月计划.md
