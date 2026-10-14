# 多元素扩展研究与设计依据

本文用于规划 Hg-MC-Auto 的元素扩展。文献条目按 DOI 元数据核对；“顶刊应用”与“分析方法论文”分开，不把期刊影响力等同于方法适用性。具体同位素比值、化学分离、校正模型和标准物质应回到论文全文及实验室验证后才写入软件配置。

## 可核实的 MC-ICP-MS 方法文献

| 系统 | 论文 | 年份 / 期刊 | DOI | 对软件设计的价值 |
|---|---|---|---|---|
| Cu、Zn | C. N. Maréchal, P. Télouk, F. Albarède, “Precise analysis of copper and zinc isotopic compositions by plasma-source mass spectrometry” | 1999, *Chemical Geology* 156, 251–273 | [10.1016/S0009-2541(98)00191-0](https://doi.org/10.1016/S0009-2541(98)00191-0) | 一个方法论文覆盖两个元素；适合研究共享分离/质量偏差校正接口，但 Cu 与 Zn 仍应有独立方法配置。 |
| Mo | C. Siebert, T. F. Nägler, J. D. Kramers, “Determination of molybdenum isotope fractionation by double-spike multicollector inductively coupled plasma mass spectrometry” | 2001, *Geochemistry, Geophysics, Geosystems* 2 | [10.1029/2000GC000124](https://doi.org/10.1029/2000GC000124) | 双尖峰、化学纯化、多个质量比内部一致性检查和外部重现性，说明不能把所有元素强行塞入同一校正公式。 |
| Hf–Lu | J. Blichert-Toft, C. Chauvel, F. Albarède, “Separation of Hf and Lu for high-precision isotope analysis of rock samples by magnetic sector-multiple collector ICP-MS” | 1997, *Contributions to Mineralogy and Petrology* 127, 248–260 | [10.1007/s004100050278](https://doi.org/10.1007/s004100050278) | Hf/Lu 展示化学分离可成为高精度流程核心；软件应记录分离批次、空白和回收率等 provenance。 |
| Ba | X. Nan et al., “High-precision barium isotope measurements by MC-ICP-MS” | 2015, *Journal of Analytical Atomic Spectrometry* 30, 2307–2315 | [10.1039/C5JA00166H](https://doi.org/10.1039/C5JA00166H) | 可作为完整方法模块的例子，需提取干扰校正、标准物质和精密度控制细节。 |
| Rb | X. Hu et al., “High precision Rb isotope measurements by MC-ICP-MS” | 2021, *Journal of Analytical Atomic Spectrometry* 36, 2744–2755 | [10.1039/D1JA00315A](https://doi.org/10.1039/D1JA00315A) | 适合作为新元素配置和参考物质/地质标准验证的模板。 |
| Sm | J.-H. Bai et al., “High intermediate precision Sm isotope measurements in geological samples by MC-ICP-MS” | 2023, *Journal of Analytical Atomic Spectrometry* 38, 629–637 | [10.1039/D2JA00412G](https://doi.org/10.1039/D2JA00412G) | “中间精密度”提醒软件需要按批次/日期/操作者跟踪 QC，而不只看单次重复测量。 |

## 高影响力应用论文（不是分析方法规范）

| 系统 | 论文 | 年份 / 期刊 | DOI | 适用说明 |
|---|---|---|---|---|
| Mo | G. L. Arnold, A. D. Anbar, J. Barling, T. W. Lyons, “Molybdenum Isotope Evidence for Widespread Anoxia in Mid-Proterozoic Oceans” | 2004, *Science* 304, 87–90 | [10.1126/science.1091785](https://doi.org/10.1126/science.1091785) | 高影响力应用案例，适合理解 Mo 同位素数据解释场景；不能代替方法论文和实验 SOP。 |

## 建议优先级

1. **先把 Hg 做成可复现基线**：分离计算、经验/ML 诊断、结构化 QC、批次审计和人工复核串成一条可重跑流程。
2. **Cu / Zn：先做离线方法插件原型**：应用场景和方法文献都较丰富；但必须由实验室提供同位素比、干扰、标准物质和校正 SOP。
3. **Mo：第二个科学验证目标**：方法差异（double-spike）明确，可检验插件是否真正支持元素专属数学模型。
4. **Ba / Rb / Sm：扩展候选**：按 JAAS 方法论文逐篇提取 QC 与精密度验证要求。
5. **Fe、Cd：保持 scaffold 状态**：当前仓库没有足够完整的元素专属分析流程，不应展示成 production-ready。

## 插件边界

每个元素方法建议由声明式 `ElementMethod` 配置描述：

- 同位素系统、测量质量数/比值与符号约定
- 采集模式、杯位/仪器方法元数据（不直接操控仪器）
- 干扰及校正方法、质量偏差/双尖峰算法选择
- 化学分离批次、空白、回收率和样品制备元数据
- 标准物质、长期控制图、精密度/准确度验收标准
- 原始输入列映射、输出字段、QC 规则及可追溯文献

核心 pipeline 应只负责：`导入 → 校验 → 计算 → QC → 报告`。元素差异由可测试的插件/方法配置承担。LLM 只解释结构化 QC 证据并给出建议，不能替代数值算法或实验 SOP；其请求必须带上数据来源与人工复核状态。

## 证据与上线门槛

文献记录不等于已实现。一个新元素在 UI 中标为 production 前，至少需要：方法文献审查、经实验室批准的 SOP、标准物质/重复样验证数据、干扰和校正的单元测试、与独立软件或人工计算的对照，以及版本化的 QC 阈值来源。
