# 基于已提供论文的医学影像近缘文献调研报告

## 执行摘要

你提供的三篇论文共同聚焦**肝细胞癌术前微血管侵犯预测**，核心模态是**CEUS/ Sonazoid-CEUS**，方法路线清晰地从**超声放射组学**演进到**视频级深度学习**与**Transformer**，并开始把**跨机构泛化**与**预后分层**纳入核心评价。综合近五年高质量原始论文后，我认为最相近、最值得对标的文献主要分为两组：一组是**Sonazoid/CEUS 影像特征或放射组学的多中心研究**，另一组是**CT/MRI 场景下的多模态、可解释或 Transformer 深度学习研究**，可作为方法迁移参照。整体上，性能上限已可到 AUC 0.84–0.94，但**跨中心性能衰减、代码公开不足、标注与采集协议不统一**，仍然是临床落地的主要瓶颈。citeturn6search1turn7search1turn0search2turn13view0turn12view0turn21view1turn35view0

## 已提供论文的主题与方法抽取

从你提供的论文可以提炼出一条很明确的研究主线。第一篇工作把**CEUS 动态视频**直接输入深度学习框架，用 **GRU 模块建模时序、CNN 模块建模纹理**，再与临床变量融合，并进一步验证其对 OS/RFS 的分层能力；这是“**从静态影像转向视频级表征**”的关键一步。第二篇工作则进一步转向 **Sonazoid-CEUS + Transformer**，强调在小肝癌（≤5 cm）中进行更强的表示学习和外部测试；第三篇工作把问题推进到**跨机构评估**层面，对比深度学习与放射组学，并直接暴露了域外性能下降这一现实问题。citeturn6search1turn7search1turn0search2

因此，后续最相近文献不应只看“同为 HCC-MVI 预测”，还应优先覆盖四个维度：**同模态（US/CEUS/Sonazoid）**、**同任务（术前 MVI 预测，最好含预后）**、**同方法谱系（radiomics→video DL→Transformer）**、**同验证范式（外部验证/多中心/读片者对比）**。按这个标准，最值得重点对读的是：Dong 2019/2020/2022、Zhou 2021、Yao 2023、Lu 2024，以及方法迁移意义很强的 Wei 2021、Wang 2023、TED 2022、MVI-TR 2023。citeturn33view0turn34view3turn3view3turn9view2turn13view0turn12view0turn26view3turn28view0turn27view1turn21view1

## 详细比较表

> 注：不同研究在**入组标准**（如单发 HCC、≤5 cm、单中心/多中心）、**验证方式**（内部、外部、前瞻性）、**模态**（B-mode、CEUS、Sonazoid、CT、MRI）和**MVI 阳性比例**上差异较大，表中数值更适合做“方法图谱比较”，不宜简单横向排名。citeturn35view0turn12view0turn26view1turn21view1

| 论文 | 年份 | 期刊/会议 | 方法/模型 | 数据集/标注 | 关键指标/结果 | 可复现性/代码链接 | 与已提供论文的相似点/差异 |
|---|---:|---|---|---|---|---|---|
| Zhang 等，*Deep Learning of Liver CEUS to Predict MVI and Prognosis in HCC*（你提供）citeturn6search1 | 2022 | Frontiers in Oncology | **CEUS 视频级 DCNN**；GRU 建模时序 + CNN 建模纹理；与临床变量融合成 CECL-DCNN | 436 例术前 CEUS；训练/验证/测试 = 301/102/33；病理为金标准 | 测试集 fusion 模型 **AUC 0.865**；特异度 81.0%，准确率 78.8%；预测 MVI 还能分层 OS/RFS | 开放全文；论文页未明确给出官方代码 | 与你的主题最直接一致：**CEUS 视频、MVI、预后**；但为**单中心回顾性**，外部泛化证据不足。citeturn6search1turn31search14 |
| Qin 等，*Transformer model based on Sonazoid CEUS for MVI prediction in HCC*（你提供）citeturn7search1 | 2025 | Medical Physics | **Transformer + ResNet-101**；基于 Sonazoid-CEUS 与原始超声图像的深度表征 | 618 例 HCC（≤5 cm），含内部测试与外部测试；与两名临床医师比较 | 最优 Transformer **内部 AUC 0.93、外部 AUC 0.84**；优于经验读片者 | 文中注明可在 entity["company","GitHub","code hosting company"] 获取代码。fileciteturn0file1 | 与你的研究最像：**Sonazoid、深度学习、外部测试、临床对照**；是当前最值得直接复现与对标的强基线之一。citeturn7search1 |
| Lu 等，*Cross-institutional evaluation of deep learning and radiomics models in predicting MVI ... using ultrasound and CEUS images*（你提供）citeturn0search2 | 2024 | Cancer Imaging | **静态/时序 US 与 CEUS 深度学习模型**，并与 radiomics、放射科医师 head-to-head 比较 | 309 例，来自 13 家机构建模；另有 2 家机构外测；病理金标准 | 内部 AUC 约 **0.73–0.84**，外部 AUC 约 **0.64–0.75**；显著暴露跨中心退化 | Open access；更强调评测框架，未在摘要页明确代码 | 这是与你当前研究最有“方法学共振”的文献：重点不是再追更高内部 AUC，而是**证明/解释外部泛化**。citeturn0search2 |
| Dong 等，*Radiomics Algorithm Based on Ultrasound Original Radio Frequency Signals* citeturn33view0 | 2019 | Frontiers in Oncology | **ORF + SAP + radiomics + SR/SVM**；超声原始射频信号而非仅灰阶图像 | 前瞻性 42 例；MVI 阳性 21 例 | 最佳 **AUC 95.01%、ACC 92.86%、SEN 85.71%、SPE 100%** | 开放全文；未见官方代码 | 是**超声 MVI 预测的奠基性工作**之一；与现有 Transformer/视频方法相比样本很小，但证明了**原始信号信息量**。citeturn33view0 |
| Dong 等，*Initial Application of a Radiomic Algorithm Based on Grayscale Ultrasound Images* citeturn34view3 | 2020 | Frontiers in Oncology | **灰阶超声 radiomics**；GTR/PTR/GPTR 特征；mRMR/RF + logistic nomogram | 322 例；两阶段任务：MVI 阴/阳 + M1/M2 分级 | GPTR radiomic signature AUC **0.726**；加 AFP 后 AUC **0.744** | 开放全文；未见官方代码 | 与你提供论文相比，方法更早、更偏 handcrafted；但其“**肿瘤周围区**”思想后来在 Sonazoid/Kupffer 期研究中被不断强化。citeturn34view0turn34view3 |
| Dong 等，*Kupffer phase radiomics features of Sonazoid CEUS: A prospective study* citeturn3view3 | 2022 | Clinical Hemorheology and Microcirculation | **Sonazoid Kupffer 期 radiomics**；MRMR + LASSO + SVM；与年龄做 logistic 融合 | 前瞻性 100 例；肿瘤区与 5 mm 周围区分割 | Kupffer 期周围组织特征 AUROC **0.800**；年龄+kupfferPT 模型 AUROC **0.804**，ACC 75.0% | 摘要可得；未见官方代码 | 与 Qin 2025 和 Lu 2024 高度接近：都强调 **Sonazoid/Kupffer 期 + 周围区**；但仍是小样本传统 ML。citeturn3view3 |
| Zhou 等，*Combining Clinical Features and CEUS LI-RADS Improves Prediction of MVI* citeturn9view2 | 2021 | Frontiers in Oncology | **CEUS LI-RADS + 临床 nomogram**；逻辑回归 | 双中心 127 例；训练 98、外测 29 | 测试集 **AUC 0.84**；灵敏度 **86.7%**，高于单纯临床模型 | 开放全文；未见官方代码 | 不是深度学习，但具有强临床可解释性；可作为你后续工作中“**可解释临床基线**”而非主模型。citeturn9view2 |
| Yao 等，*Analysis of Sonazoid CEUS for predicting MVI risk: a prospective multicenter study* citeturn13view0 | 2023 | European Radiology | **Sonazoid-CEUS 影像生物标志物 + 临床联合模型** | 前瞻性多中心 211 例；导出 170、外部验证 41 | 联合模型 AUROC **0.859/0.812**（导出/外部验证） | 期刊官网摘要可得；代码未明确 | 这是你提供论文之外，**Sonazoid 多中心前瞻性**证据最重要的一篇；特别适合作为 Qin 2025 之前的强“传统特征”参照。citeturn13view0 |
| Lu 等，*Prediction of MVI with conventional US, Sonazoid-CEUS, and biochemical indicator* citeturn12view0 | 2024 | Insights into Imaging | **B-mode + Doppler + Sonazoid Kupffer 期 + PIVKA-II** 联合 logistic/nomogram | 三中心 318 例；开发集 246，独立外测 72 | validation/test AUC **0.937/0.893**；校准良好；DCA 显示净获益 | Open access；数据可按需申请；代码未明确 | 与你提供的 Sonazoid-Transformer 论文在**临床场景**上最接近；说明即使不用 DL，**多模态超声 + 生物标志物**也很强。citeturn12view0 |
| Wei 等，*Prediction of MVI via Deep Learning: A Multi-Center and Prospective Validation Study* citeturn26view3 | 2021 | Cancers | **CE-CT / EOB-MRI 深度学习 + attention map**； head-to-head 多模态比较 | 五家三级医院共 750 例；外部前瞻性验证 115 例 | 验证集 **CT AUC 0.736，EOB-MRI AUC 0.812**；EOB-MRI 优于 CT | Open access，含补充材料；未见官方代码 | 虽非超声，但在“**多中心、前瞻性、双模态比较**”上非常权威，适合作为你未来临床验证设计模板。citeturn26view1turn26view3 |
| Wang 等，*A novel multimodal deep learning model ... based on MRI and CT* citeturn28view0 | 2023 | European Journal of Surgical Oncology | **ResNet18 CT+MRI 多模态 DL**；再与临床/放射学组合 | 397 例；训练 297、验证 100；多中心回顾性 | **DL CT+MRI AUC 0.819**，优于单模态 DLCT 0.742；可分层 RFS | 摘要可得；代码未明确 | 方法上与 Zhang 2022 的“**影像+临床融合**”同谱系，只是模态换成 CT/MRI；证明多模态融合确实带来增益。citeturn28view0 |
| Cao 等，*MVI-TR: A Transformer-Based DL Model with CECT* citeturn21view1 | 2023 | Cancers | **Transformer（MSA+MLP）**；与 ResNet18/50/101、对比学习比较；交叉熵、label smoothing、drop path | 559 例；训练 448、验证 111 | 验证集 **AUC 0.935，ACC 0.972，Precision 0.973，Recall 0.931，F1 0.952** | Open access；论文页未明确代码 | 这是 Qin 2025 在方法上最接近的跨模态参照：说明 **Transformer 在 MVI 预测中可显著优于 CNN 基线**。citeturn20view0turn21view1 |

## 关键论文要点提炼

**Zhang 2022（你提供）**：这篇论文的关键贡献不只是做出 CEUS-MVI 分类器，而是把**视频时序**与**预后分层**连在一起，证明“从静态相位图像转向完整 CEUS 视频”是有临床价值的。它非常适合做你后续工作中“视频级方法”的基础 benchmark。citeturn6search1turn31search14

**Qin 2025（你提供）**：这是当前与你目标最接近的“**Sonazoid + Transformer + 外部测试**”工作。其价值在于：一方面验证了 Transformer 在 HCC≤5 cm 场景的有效性；另一方面已经给出代码入口，具备最强的直接复现潜力。citeturn7search1 fileciteturn0file1

**Cross-institutional 2024（你提供）**：这篇文章的重要性在于它把研究重点从“内部高分”转移到“**跨机构是否掉点**”。如果你未来要做有说服力的临床转化工作，这篇应被视作核心参照，因为它直接说明域外评价才是真问题。citeturn0search2

**Dong 2022**：它证明了 **Sonazoid Kupffer 期的肿瘤周围组织信息**对 MVI 很关键，这个结论与 2024–2025 的 Sonazoid 深度学习路线高度一致。即使模型较传统，这篇仍是“为什么 Sonazoid 值得做”的机制型证据。citeturn3view3

**Yao 2023**：这是 Sonazoid-CEUS 在**前瞻性多中心**场景中最有代表性的论文之一。它说明：即便不使用深度学习，只要把 Sonazoid 的关键影像 biomarker 建模好，也能稳定做出外部验证 AUC 0.8+ 的模型。citeturn13view0

**Lu 2024**：它把 **B-mode、Doppler、Sonazoid-Kupffer 期与 PIVKA-II** 系统性整合，得到验证/测试 AUC 0.937/0.893。对你的研究最有启发的是：**深度学习不一定是唯一强解，多模态超声+生物标志物联合基线必须纳入。**citeturn12view0

**Wei 2021**：这篇多中心、前瞻性、CT 与 EOB-MRI head-to-head 的设计很强，虽然不是超声，但在**验证方案**上非常值得借鉴。它提示：方法学上真正高质量的临床 AI 研究，必须把**模态比较、外部验证和注意力可视化**一起做。citeturn26view1turn26view3

**MVI-TR 2023**：这是“Transformer 迁移到 MVI 预测”的强参照。它不只是把 AUC 提高到 0.935，还系统报告了与 ResNet、对比学习模型的对照、校准和 DCA，这种完整报告方式非常值得你后续论文结构直接借鉴。citeturn20view0turn21view1

## 对比分析与建议

把这些文献放在一起看，有三个高置信结论。

第一，**Sonazoid/Kupffer 期与肿瘤周围区信息是超声路线最稳定的增益来源**。从 Dong 2022 的 kupfferPT、到 Yao 2023 的灰度比/洗脱时间、到 Lu 2024 的 Kupffer 期清除模式，再到 Qin 2025 的 Sonazoid-Transformer，信号都指向同一件事：MVI 并不只存在于肿瘤内部纹理，而更强烈地体现在**肿瘤边界—周围肝组织—Kupffer 相互作用**上。citeturn3view3turn13view0turn12view0turn7search1

第二，**时序建模与多模态融合确实优于纯静态或纯临床基线，但“外部泛化”才是最终裁判**。CEUS-DCNN 视频模型、CT+MRI 多模态 DL、Transformer 模型都能在内部或单中心验证中取得较高 AUC；但你提供的 2024 跨机构评估已经清楚表明，域外性能会明显下降。这个结论与多中心/前瞻性研究的经验完全一致。citeturn31search14turn28view0turn21view1turn0search2turn26view1

第三，**公开可复现性仍明显落后于性能提升速度**。以超声路线为例，已有论文可以达到或接近 AUC 0.84–0.89，部分单中心或小样本研究甚至更高；但系统综述显示，超声放射组学的汇总 AUC 约为 **0.81**，说明真实世界可迁移性能仍受异质性限制。再加上多数论文并未在官方页面明确给出可直接运行的代码，复现实用门槛偏高。citeturn35view0turn12view0turn13view0turn7search1

基于这些发现，我对你下一步研究的建议是：

其一，**把对照基线做“强”**。最低应同时包含：  
临床模型；传统 radiomics（tumor / peritumor）；B-mode-only；CEUS 静态相位模型；CEUS 视频模型；Sonazoid-专用模型；以及“影像+生物标志物”联合模型。否则即使新模型分数高，也无法说明增益来自哪里。这个建议直接来自 Lu 2024、Zhang 2022 和你提供的跨机构研究。citeturn12view0turn6search1turn0search2

其二，**把泛化实验前置到设计阶段，而不是发表前补一个“外测”**。最合适的方案是：按中心划分数据，保留至少一个**锁定外部测试集**，再做 **leave-one-center-out** 或时间外推验证；同时在不同机器厂商、不同对比剂注射与相位采样策略下做亚组分析。因为 2024 跨机构研究已经证明，仅靠随机切分无法代表真实部署表现。citeturn0search2turn17view0

其三，**把可复现性包做完整**。建议至少同步公开：固定的数据划分文件、预处理脚本、训练配置、最佳权重、推理脚本、病例级而非切片级评价代码，以及临床变量字典。若数据不能完全公开，至少应公开匿名化示例、特征表结构与推理接口。最现实的发布位置仍是 entity["company","GitHub","code hosting company"]。Qin 2025 已经展示了这一点对复现价值的提升。citeturn7search1 fileciteturn0file1

其四，**把临床验证终点从单一 AUC 扩展到校准、决策曲线、读者比较和预后**。如果你的研究确实面向术前决策，那么除了 ROC/AUC，还应固定报告：校准曲线、Brier score、DCA、在固定特异度下的灵敏度、与资深/初级医师的 head-to-head，以及对 RFS/OS 的风险分层。这方面 Zhang 2022、Lu 2024、MVI-TR 2023 和 Wei 2021 都给出了很好的模板。citeturn6search1turn12view0turn20view0turn26view1

如果你的目标是做一项**可用于临床验证的下一步实验**，我建议采用如下最小可发表设计：  
选择 ≥3 家中心，连续入组单发 HCC（可分 ≤5 cm 与 >5 cm 两层）；采集 B-mode、原始 CEUS 视频、Sonazoid Kupffer 期关键帧、基础实验室指标（AFP/PIVKA-II 等）；以病理为金标准；主终点设为**外部中心病例级 AUC 与校准**，次终点设为**读者比较、RFS 分层、亚组稳定性**；模型上同时报告 radiomics、GRU/CNN 视频模型、Transformer、域泛化版本。这样得到的证据强度，会明显高于单纯“新网络 + 随机切分”。citeturn0search2turn12view0turn13view0turn26view1turn21view1

## 检索策略与开放问题

本次检索以你提供论文自动抽取出的主题为起点，优先检索以下关键词组合：**hepatocellular carcinoma / HCC, microvascular invasion / MVI, ultrasound, contrast-enhanced ultrasound / CEUS, Sonazoid, Kupffer phase, radiomics, deep learning, transformer, video, external validation, multicenter, prognosis**。来源优先级为：**原始论文官方页面、PubMed 摘要、开放获取期刊全文、官方补充材料与明确给出的代码入口**。citeturn6search1turn7search1turn12view0turn13view0turn21view1

仍需保留的开放问题主要有三点。第一，部分高影响力论文仅能获取摘要页，像损失函数、超参数、确切数据划分和代码细节并不总是公开。第二，不同研究对纳入标准和 MVI 分层的定义并不完全一致，导致数值难以绝对横比。第三，当前超声路线虽然已经有不错的外部结果，但系统综述仍显示异质性存在，说明“**跨设备、跨中心、跨操作者**”的一致性问题尚未真正解决。citeturn35view0turn12view0turn0search2

如果你后续要把这份报告转成“论文综述引言”或“related work”章节，最建议优先引用的主干文献序列是：**Dong 2022 → Yao 2023 → Lu 2024 → Zhang 2022 → Qin 2025 → Cross-institutional 2024**；若要强调方法创新，再补 **Wei 2021、Wang 2023、TED 2022、MVI-TR 2023**。这能同时覆盖**超声任务相似性、方法演进、以及临床验证强度**三个维度。citeturn3view3turn13view0turn12view0turn6search1turn7search1turn0search2turn26view1turn28view0turn27view1turn21view1