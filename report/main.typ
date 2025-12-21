#import "./template.typ": *

// page & layout
#show: doc => setup-base-fonts(
  doc,
  first-line-indent: 0em, 
)
#set page(
  paper: "a4",
  margin: (top: 22mm, bottom: 22mm, left: 22mm, right: 22mm),
)

// title-page setup
#show: labreport.with(
  logos: (image("inc/logo.png"), none),

  report-title: "目标检测模型的训练（期末报告）",
  exper-name: "基于Attention R2U-Net和YOLO的多阶段\n晶圆芯片分割（语义分割）和\n缺陷检测（目标检测）",

  course-name: "深度学习及其应用",
  class: "CS40034.01",
  faculty: "软件学院",

  student-no: "22302016002",
  student-name: "王俊崴（成员：江琦琦，洪图）",

  exper-date: "2025-12-13",
  handin-date: datetime.today().display(),
)

#set par(
  first-line-indent: (amount: 2em, all: true),
  spacing: 1.5em,
)

// table of contents
#show: cellpress.style-table

== 项目仓库

- 训练代码与实验记录：#link("https://github.com/jwwang2003/deep-learning-pj")[https://github.com/jwwang2003/deep-learning-pj]
- 集成应用（部署/可视化）：#link("https://github.com/jwwang2003/aoi-wafer-stacking")[https://github.com/jwwang2003/aoi-wafer-stacking]

#pagebreak(weak: true)
#outline(title: "目录")
#pagebreak(weak: true)

// main

= 引言

什么是自动光学检测（AOI，Automated Optical Inspection）？
它是一种利用机器视觉与图像处理技术，对工业产品进行非接触式外观与结构检测的自动化检测手段。
它通过高分辨率相机获取产品表面或内部的图像信息，并结合算法对图像进行分析，从而判断是否存在缺陷。
AOI广泛应用于半导体、电子制造、PCB、面板和新能源器件等领域，尤其适合对精度、稳定性和一致性要求极高的生产环境。

AOI之所以被广泛使用，主要是因为传统人工检测效率低、主观性强，且难以满足现代制造业高速、大批量、高良率的生产需求。
在实际应用中，AOI系统通常部署在产线关键工序节点，对器件进行实时或准实时检测，可快速识别诸如划伤、污染、缺口、对位偏差等缺陷。
随着深度学习的发展，AOI已从早期的规则与阈值判断，逐步演进为结合卷积神经网络的智能检测系统，大幅提升了对复杂缺陷和微小异常的识别能力，为提升产品质量、降低返工率和推动智能制造发挥了重要作用。


针对大尺寸晶圆级 die 图像的自动光学检测，不仅需要准确判断是否存在缺陷（命中 / 未命中），还需要具备精确的空间推理能力。
本项目围绕这一需求，设计并实现了两个相互耦合的实验方向：
- （1）基于 Ultralytics YOLOv5，并进一步升级至 YOLO11/YOLO12 的四分类缺陷目标检测模型微调实验；
- （2）训练一个定制化的 Attention R2U-Net（ATTR2UNet）分割模型，用于对一个图片里的单个 die 进行精确分割，从而使下游逻辑能够判断缺陷是否落在有效芯片区域内防止误判，以及缺陷在芯片内部的具体位置分布。

实验结果表明，在面向实际生产与工程应用的 AOI 场景中，检测系统的可信度不仅取决于模型的数值指标，还高度依赖于成像条件、误差分析方法以及具体的应用场景与工艺背景。
仅凭模型精度或分数并不足以支撑面向生产的可靠判断，这一点也构成了本项目在课程设计中强调工程完整性与系统可靠性的核心认识。

== 目标

本次实践的目的是搭建一个AOI的检测系统，且包含以下几个任务：
- 建立一个 可复现的 YOLOv5 基线模型用于 AOI 数据集，并量化升级至 YOLO11 / YOLO12 所带来的性能提升；
- 训练一个 Attention R2U-Net 分割模型，生成稳定可靠的 die（芯片）掩膜，从而支持对 芯片归属关系（die ownership） 及 芯片内部缺陷区域分布 的分析与推理；
- 系统分析 数据、模型以及整体检测流程集成 中存在的不足，并提出针对性的改进方案，以提升系统的 整体稳定性与可靠性。

#pagebreak(weak: true)

== 术语表（Vocabulary）

#table(
  columns: (1fr, 1fr),
  inset: 6pt,
  stroke: (paint: luma(40%), thickness: 0.8pt),
  align: (left, left),
  [*English*], [*中文*],
  [AOI (Automated Optical Inspection)], [自动光学检测],
  [Wafer], [晶圆],
  [Die], [芯片单元],
  [Defect], [缺陷],
  [Semantic segmentation], [语义分割],
  [Object detection], [目标检测],
  [Mask], [掩膜],
  [Process drift], [工艺漂移],
  [heterogeneity], [异质性]
)

#pagebreak(weak: true)

= 方法

== 数据集的准备

在基于深度学习的 AOI 缺陷检测任务中，数据集质量直接决定模型性能上限与系统可靠性。相比算法结构本身，数据是否真实、全面、具有代表性，往往对模型在实际产线中的泛化能力影响更大。因此，在构建 AOI 数据集时，不仅需要覆盖常见缺陷类型，还必须充分考虑成像条件、工艺波动以及产线差异等现实因素。

本项目使用的 AOI 数据集由多批次晶圆/芯片（die）图像构成，尽可能还原真实生产环境中的复杂情况。由于采集来源和时间不同，数据在多个维度上存在显著差异：
- 光照条件变化：部分样本曝光偏暗、对比度不足，而部分样本光照均匀清晰；
- 工艺漂移（process  drift）：不同批次晶圆在纹理、背景和边缘形态上存在系统性差异；
- 版图与间距变化：不同运行批次中 die 的排布方式和间距并不完全一致。

这些因素使得数据集呈现出明显的异质性（heterogeneity），同时也构成了模型泛化与稳定性评估的重要基础。@fig-appearance 展示了不同成像条件和数据分布下的典型晶圆裁剪示例。

#subpar.grid(
  figure(image("images/examples/dark.png"), caption: [
    曝光过暗
  ]), <a>,
  figure(image("images/examples/normal.png"), caption: [
    正常曝光
  ]), <b>,
  figure(image("images/examples/odd.jpg"), caption: [
    位移或不对称
  ]), <c>,
  figure(image("images/examples/different.png"), caption: [
    颜色尺寸差异
  ]), <d>,
  columns: (1fr, 1fr, 1fr, 1fr),
  caption: [
    #strong[晶圆裁剪示例。]\
    @a：难以识别、光照偏暗和对比度低的边缘缺陷；\
    @b：正常光照采集样本，用于基线模型的超参数调优；\
    @c，@d：“差异”数据集，色调偏暖且存在轻微镜头畸变；\
    用于扩展和测试模型的域泛化能力。
  ],
  label: <fig-appearance>,
)

我们的数据集规模较大：同一芯片型号对应多批次缺陷采样，每个批次均包含上千张图片。受数据量庞大以及人力、时间成本限制，目前我们完成的人工标注数据约一千余张。其中，目标检测标注数据约 400 张；语义分割标注数据超过 700 张。之所以语义分割数据量更大，是因为语义分割模型并非在预训练模型上进行简单微调，而是需要从零开始训练网络，对数据规模的要求更高。

在上述数据规模与标注需求的约束下，如何高效、规范地完成标注工作，并保证标注结果的可复现性与可扩展性，成为本项目中的关键问题。因此，我们在数据标注阶段引入了统一的标注工具与流程，具体如下。

=== 数据标注软件

为保证后续模型训练的数据来源清晰、标注过程可追溯，我们将人工标注流程做成标准化、可复现的流水线：统一使用同一套标注工具、固定的数据目录结构、明确的导入/导出格式与版本管理策略。基于此，下面介绍本项目的数据标注软件与具体操作流程。

为保障标注流程可复现，本项目采用 Label Studio@label_studio 并使用 Docker 方式部署。其优势在于：支持多用户与多项目管理、可配置标注界面模板、导出格式丰富（便于适配检测/分割训练管线），同时数据与标注结果可通过挂载目录实现持久化，方便版本管理与复现实验。

+ 启动与持久化目录（建议的目录结构）\
为避免容器重启导致数据丢失，我们将 Label Studio 的工作目录整体挂载到本地，例如：\
  - label-studio-data/：Label Studio 的数据库与上传文件等资产（持久化核心）\
  - datasets/：原始图像数据（只读或独立管理，避免与标注资产混在一起）\
  - exports/：导出的标注文件（按任务类型/日期/版本归档）\
  一个推荐的结构示例：\
    ```text
      project_root/
        datasets/
          det_images/
          seg_images/
        label-studio-data/
        label_configs/
          det_config.xml
          seg_config.xml
        exports/
          det/
          seg/
    ```

#set enum(start: 2)
+ Docker 启动命令（单机快速部署）

```bash
docker run -it -p 8080:8080 \
  -v "$(pwd)/label-studio-data":/label-studio/data \
  --name label-studio \
  heartexlabs/label-studio:latest
```

首次启动后访问 `http://localhost:8080` 完成账号创建；项目数据与导出文件会保存在 `label-studio-data` 目录，便于版本管理与复现。若需要更详细的日志以便排查问题，可在镜像启动命令末尾追加参数（如调高日志级别）。

#set enum(start: 3)
+ 项目创建与数据导入
- 在 Web 界面创建项目（建议分别创建“目标检测项目”和“语义分割项目”，避免标签体系混用）。
- 导入图像数据：可以直接上传文件，或导入本地/云端存储（取决于团队数据管理方式）。
- 导入标注配置模板（Label Config）：将本项目使用的配置文件（如 det_config.xml、seg_config.xml）纳入版本管理（Git），确保每次标注界面一致，从而保证可复现性。

#set enum(start: 4)
+ 标注执行与质量控制
- 标注规范：统一类别命名、缺陷边界定义与“难例/不确定样本”的处理规则（例如设置 ignore 或 uncertain 标签）。
- 过程控制：按批次分配任务（Batch/型号/工艺段），每批抽检一定比例样本进行复核；必要时进行二次标注或一致性检查。
- 记录策略：在项目说明中记录标签版本、标注人员、抽检比例与修订时间，保证数据可追溯。

#set enum(start: 5)
+ 导出与下游训练对接
完成标注后，从 Label Studio 导出标注结果，并按任务类型归档：
- 目标检测：导出包含框（bounding box）信息的标注文件，用于后续转换为训练所需格式（如 COCO/YOLO 等，具体以训练代码要求为准）。
- 语义分割：导出分割标注（mask/多边形等）对应的结果，用于生成像素级标注（mask）或统一的分割标注格式。
#set enum(start: 1)

导出文件建议按“任务类型 + 数据版本号 + 导出日期”命名并存放在 exports/ 下，确保训练时可以精确定位“用了哪一版标注数据”。

通过上述流程，我们获得了可追溯、可复现的检测与分割标注数据。接下来将对导出标注进行格式整理与转换，并构建统一的数据集目录与加载流程，以支撑后续模型训练与评估。

=== 缺陷的类型

在对数据进行系统分析后，我们明确了本任务中需要重点关注和检测的缺陷类型。结合实际 AOI 场景与标注可行性，最终定义了以下五个类别：
- 崩边（Edge chip）：芯片边缘存在破损或缺口，可能在后续封装或使用中引发可靠性问题；
- 墨点（Ink spot）：局部黑色点状残留，通常由污染、印记或工艺残留造成；
- 沾污（Stain）：芯片表面附着异物或脏污，可能遮挡有效结构并影响电性或外观判断；
- 其他缺陷（Other defect）：不易归入上述类别但明显异常的缺陷，用于增强模型对未知或少见缺陷的鲁棒性；
- 可接受（Acceptable）：存在轻微外观变化但不影响功能、可放行的样本，用于区分“异常但合格”与“真实缺陷”。

@fig-types-of-defects 给出了其中四类典型缺陷（崩边、墨点、沾污、斑点）的局部裁剪示例，为后续检测与分析提供了直观参考。

#subpar.grid(
  figure(
    image("images/examples/崩边.png", width: 100%),
    caption: [崩边]
  ), <fig-types-of-defects.a>,
  figure(
    image("images/examples/墨点.png", width: 100%),
    caption: [墨点]
  ), <fig-types-of-defects.b>,
  figure(
    image("images/examples/沾污.png", width: 100%),
    caption: [沾污]
  ), <fig-types-of-defects.c>,
  figure(
    image("images/examples/斑点.png", width: 100%),
    caption: [斑点]
  ), <fig-types-of-defects.d>,
  columns: (1fr, 1fr, 1fr, 1fr),
  caption: [
    #strong[典型缺陷类型示例。]\
    图中展示了 AOI 数据集中四类常见芯片缺陷的局部裁剪图：\
    （a）崩边：边缘破损/缺口，可能影响器件可靠性与封装良率；\
    （b）墨点：局部黑色点状残留，常见于污染或印记；\
    （c）沾污：表面异物附着或脏污覆盖，可能遮挡结构并引入误检；\
    （d）斑点：非均匀点状纹理异常，反映表面/工艺异常或颗粒残留。
  ],
  label: <fig-types-of-defects>,
)

=== 缺陷分析

在标注与建模之前，对缺陷进行充分分析是必要步骤。一方面，不同缺陷在尺度、形态和位置分布上差异显著：例如崩边通常位于 die 边缘，而墨点和沾污多出现在芯片内部；另一方面，部分缺陷在视觉表现上与背景纹理或工艺噪声高度相似，容易引发误检或漏检。

在数据标注阶段，我们采用 Label Studio 完成所有样本的人工标注（见 @label_studio）。标注以边界框（bounding box）形式导出，用于目标检测任务的训练；同时，针对 die 分割任务，构建了两套 COCO 风格数据集（collection0 与 collection1）。为缓解数据异质性带来的影响，在训练阶段通过 helpers/coco_dataset.py 对图像与标注同步施加强数据增强（翻转、仿射变换、抖动、模糊等），以提升模型对光照变化、工艺漂移和版图差异的适应能力。

#subpar.grid(
  figure(
    image("images/screenshots/labelstudio/lb_main.png"),
    caption: [标注项目列表]
  ), <fig-label-studio.a>,
  figure(
    image("images/screenshots/labelstudio/lb_main2.png"),
    caption: [标注任务界面]
  ), <fig-label-studio.b>,
  figure(
    image("images/screenshots/labelstudio/lb_object.png", width: 100%),
    caption: [目标检测标注示例]
  ), <fig-label-studio.c>,
  figure(
    image("images/screenshots/labelstudio/lb_segmentation.png", width: 100%),
    caption: [分割标注示例]
  ), <fig-label-studio.d>,
  columns: (30%, 30%),
  caption: [
    #strong[Label Studio 标注流程与示例。]\
    从项目任务管理、标注界面到检测/分割样本，展示了数据准备阶段的主要标注工作流。
  ],
  label: <fig-label-studio-demo>,
)

为确保分割与检测任务的标注逻辑清晰一致，Label Studio 分别配置了分割与目标检测的界面模板，关键片段如下所示。

#subpar.grid(
  box(
    inset: 8pt,
    stroke: (paint: luma(40%), thickness: 0.8pt),
    radius: 4pt,
  )[```xml
<View>
  <Header value="Select label and click the image to start"/>
  <Image name="image" value="$image" zoom="true"/>

  <RectangleLabels name="label" toName="image"
                   strokeWidth="3" pointSize="small"
                   opacity="0.4">
    <Label value="DIE" background="red"/>
  </RectangleLabels>

  <PolygonLabels name="poly" toName="image"
                 strokeWidth="3" pointSize="small"
                 opacity="0.4">
    <Label html="DIE(poly)" value="DIE_p" background="red"/>
  </PolygonLabels>
</View>
```],
  box(
    inset: 8pt,
    stroke: (paint: luma(40%), thickness: 0.8pt),
    radius: 4pt,
  )[```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="其他外观不良" background="#FFA39E"/>
    <Label value="可接受" background="#D4380D"/>
    <Label value="崩边" background="#FFC069"/>
    <Label value="沾污" background="#AD8B00"/>
    <Label value="墨点" label="Ink Stain" background="#D3F261"/>
  </RectangleLabels>
</View>
```],
  columns: (1fr, 1fr),
  caption: [
    #strong[Label Studio 标注界面配置片段。]\
    左侧为分割任务配置（矩形 + 多边形 DIE 标注），右侧为检测任务的多类别矩形标注。
  ],
  label: <fig-label-studio-config>,
)

总体而言，本数据集的设计与标注不仅服务于模型训练本身，更强调工程可用性与真实场景覆盖，为后续 YOLO 系列检测模型与 Attention R2U-Net 分割模型的联合应用奠定了可靠基础。

== 检测流程\（Detection pipeline\）

为便于对整体系统的理解，图 @fig-aoi-pipeline 展示了 AOI 任务中分割与检测的联合流程：输入晶圆图像后，分割模型给出 die 掩膜，检测模型给出缺陷候选框，随后在 die 级别进行归属与位置推理，输出最终的工艺质检结果。

#let stage(label, desc, fill: luma(96%), stroke: luma(35%)) = {
  align(center)[
    #box(
      width: 3.4cm,
      inset: 1em,
      fill: fill,
      stroke: (paint: stroke, thickness: 0.9pt),
      radius: 5pt,
      align(center)[
        #text(weight: "bold")[#label] \
        #smallcaps[#desc]
      ]
    )    
  ]
}

#figure(
  caption: [
    #strong[AOI 分割 + 检测联合流程图。]\
    分割与检测共享原始输入；芯片掩膜为缺陷定位提供结构先验，逻辑层完成 “是否在芯片内” 与 “处于何区域” 的判断。
  ],
  grid(
    columns: 7,
    align: center + horizon,
    gutter: 6pt,

    stage("输入", "芯片图像", fill: luma(94%), stroke: luma(40%)),
    [#text(size: 20pt, weight: "bold")[→]],

    stage("ATTR2UNet", "芯片掩膜", fill: rgb("#E9F0FF"), stroke: rgb("#3B6BD6")),
    [#text(size: 20pt, weight: "bold")[→]],

    stage("YOLOv5/11", "缺陷框", fill: rgb("#FFF2E8"), stroke: rgb("#D46B08")),
    [#text(size: 20pt, weight: "bold")[→]],
    
    stage("芯片逻辑", "归属 + 区域", fill: rgb("#E9F7EF"), stroke: rgb("#2F855A")),
  ),
) <fig-aoi-pipeline>

=== 语义分割 \（Segmentation\）

语义分割以 U-Net (2015)@ronneberger2015unetconvolutionalnetworksbiomedical 为基底：编码器通过“Conv → Conv → Downsample”逐层扩大感受野提取上下文，解码器通过“Upsample → Conv → Conv”恢复空间分辨率；每一尺度的跳跃连接将高分辨率细节拼接回解码端，保证边界不会在下采样中丢失。其直观的块级理解是：
- Conv block 提取局部纹理与边缘
- Downsample 聚合更大范围上下文（“what”）
- Upsample 恢复空间位置（“where”）
- Skip connection 补回细节纹理与边界

在此基础上，主流增强路径包括：
- R2U-Net (2018)@alom2018recurrentresidualconvolutionalneural：用 Recurrent Residual Convolutional block 替代“普通卷积块”。残差项 F(x)+x 保障梯度流动；共享权重的递归卷积在 t=1..T 中反复细化同尺度特征，表达更强但带来额外时延。
- Attention U-Net (2018)@oktay2018attentionunetlearninglook：在跳跃连接前加入 Attention Gate，使用解码端的门控信号 g 引导高分辨率特征 x，仅通过与当前目标相关的细节，降低背景纹理引发的误检。
- Attention R2U-Net@alom2018recurrentresidualconvolutionalneural：在 U-Net 框架中同时使用 R2 块与注意力门控，兼顾特征迭代细化与背景抑制。

面向晶圆 AOI 的 die 分割，图样高度重复且缺陷细小，注意力门控通常是最高性价比的增强；递归步数 T>1 则更适合对细微边界进行多次精炼，但计算成本与过拟合风险也更高。基于这一权衡，本项目采用自定义的 Attention R2U-Net（ATTR2UNet）作为 die 分割主干@alom2018recurrentresidualconvolutionalneural，并设置 t=2，以在边界稳定性与吞吐间取得折中。图 @fig-attr2unet-arch 展示了网络结构。

#figure(
  image("nn/attr2unet/att-r2unet.svg", width: 90%),
  caption: [
    #strong[ATTR2UNet 模型结构示意。]\
    编码端负责多尺度特征提取，解码端逐层恢复空间分辨率，并通过注意力门控选择性融合跳跃连接特征。
  ]
) <fig-attr2unet-arch>

=== 目标检测 \（Object \(defect\) Detection\）

YOLO（You Only Look Once）属于单阶段检测器：一张图只前向一次，就直接回归候选框与类别。经典流程可拆成 Backbone → Neck → Head：Backbone 负责多尺度特征提取，Neck 融合不同分辨率的语义信息，Head 在多尺度特征图上输出分类、置信度与边界框回归，再通过 NMS 得到最终结果。

*YOLOv5（Ultralytics 实现）* 通常概括为 CSPDarknet 主干 + PANet 特征融合 + anchor-based 检测头。CSP 结构减少冗余计算并稳定训练；PANet 负责上下路径融合；anchor-based 头需要预设锚框并做匹配与调参。

*YOLO11（Ultralytics）* 仍沿用 Backbone→Neck→Head 的范式，但核心变化是更高效的主干/颈部设计与 anchor-free 检测头，并常配合解耦 head（分类/回归分支分开）以改善收敛与精度-速度权衡。对工程侧而言，anchor-free 减少了 anchor 匹配与超参调优的负担，迁移到新数据集更省事。

*未来版本（以 YOLO12 为例）* 更偏研究导向，Ultralytics 在文档中强调其注意力机制与实验性特征，但也提示稳定性与部署成本可能不如 YOLO11 友好。此类版本更适合作为研究与探索路线，而非默认工程基线。

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  stroke: (paint: luma(40%), thickness: 0.8pt),
  align: (left, left, left),
  [*版本*], [*核心结构要点*], [*工程取向*],
  [YOLOv5], [CSPDarknet + PANet + anchor-based head], [成熟稳定、生态完善，但需要 anchor 超参调优],
  [YOLO11], [更高效 backbone/neck + anchor-free + 解耦 head], [更易迁移、速度/精度平衡较好],
  [YOLO12], [注意力导向、研究特性更强], [可能更吃资源、稳定性待验证],
)

实践选择上：需要稳定和可部署时优先 YOLO11；对 anchor/NMS 经验丰富且复用旧链路时 YOLOv5 仍可靠；若用于研究或探索注意力检测，可尝试 YOLO12 路线，但需预留训练稳定性与显存成本。

#subpar.grid(
  figure(
    image("nn/yolov5/image.png", width: 100%),
    caption: [YOLOv5 架构]
  ), <fig-yolo-arch.a>,
  figure(
    image("nn/yolo11/image.png", width: 100%),
    caption: [YOLO11 架构]
  ), <fig-yolo-arch.b>,
  columns: 2,
  rows: 1,
  caption: [
    #strong[YOLO 系列检测架构示意。]\
    backbone 提取特征，neck 融合多尺度语义，检测头在不同尺度上预测类别与边界框。
  ],
  label: <fig-yolo-arch>,
)

=== 芯片级推理

下游推理（die-level reasoning）的目标，是把“检测模型给出的缺陷框 dets”与“分割/版图得到的每颗芯片掩膜 masks”对齐，从而回答两个工程问题：

1. *缺陷是否在芯片内？*
   对每个缺陷框 `det.box` 与所有 `die_mask` 计算 IoU，取最大值 `best_iou`。
   - 若 `best_iou ≥ iou_th`（例如 0.2），则认为该缺陷属于某颗 die（on-die），并绑定 `die_id`；
   - 否则标记为 *off-die*（不在芯片上，常见于划片道、背景杂质、或分割漏检导致的匹配失败）。

2. *缺陷位于芯片的哪个区域？*
   将检测框中心点 `center(det.box)` 映射到该 die 的外接矩形坐标系中做归一化，得到 `norm_center ∈ [0,1]×[0,1]`。
   然后按两种常用规则输出区域标签：
   - 同心环（rings）：中心 60% / 布线区 30% / 保护环 10%（越靠边越容易是崩边、外缘、划片影响）
   - 3×3 网格（grid3x3）：给出更细的空间定位（例如左上、中心、右下等）

*置信度校准仍未完全解决*：因此报告中同时保留 检测置信度 `det.score` 与 die 掩膜概率/质量指标 `die.prob`（若有），便于人工复核与后续校准。

#show: style-algorithm
#algorithm-figure(
  "Die-level reasoning",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *

    Comment[
      说明：\
      - dets: 检测输出列表，每个 det 至少包含 det.box, det.score\
      - masks: die 掩膜列表，每个 mask 至少包含 mask.id, mask.bounding_rect（外接矩形）, (可选) mask.prob\
      - iou_th: on-die 判定阈值（常见 0.1~0.3，示例 0.2）\
      - region_mode: "rings" 或 "grid3x3"
    ]

    Procedure(
      "Die-Level-Reasoning",
      ("dets", "masks", "iou_th", "region_mode"),
      {
        Comment[Initialize output list]

        For(
          "det in dets",
          {
            Comment[(1) Select die mask with maximum IoU]

            Assign[best_iou][0]
            Assign[best_die][*null*]
            For(
              "mask in masks",
              {
                Assign[iou][#FnInline[IoU][det.box, mask]]
                If(
                  "iou > best_iou",
                  {
                    Assign[best_iou][iou]
                    Assign[best_die][mask]
                  },
                )
              },
            )
            Comment[(2) Decide on-die vs off-die using threshold]
            If(
              "best_iou < iou_th",
              {
                Comment[Keep score + off-die status for debugging]
                Assign[det.status][off-die]
                Return[det.status]
              },
            )
            Assign[det.status][on-die]
            Assign[det.die_id][best_die.id]
            
            Comment[(3) Normalize location and assign region label]
            Assign[center][#FnInline[center][det.box]]
            Assign[norm_center][#FnInline[normalize][center, best_die.bounding_rect]]
            ElseIf(
              "region_mode == rings",
              {
                Assign[det.region][#FnInline[assign_ring][norm_center]]
              },
              {
                Assign[det.region][#FnInline[assign_grid_3x3][norm_center]]
              },
            )

            Comment[(4) Report: include det.score and optional best_die.prob for review]
            Assign[det.report][(score, die_id, region)]
          },
        )
      },
    )
  }
)

== 方法小结

本章围绕 AOI 场景的分割与检测协同流程，给出了数据准备、模型结构与推理逻辑的整体框架。通过 die 掩膜与缺陷检测的联合使用，系统能够在保持效率的同时提供更可解释的空间定位依据。接下来的训练部分将补充具体实验设置与实现细节。

#pagebreak(weak: true)

= 训练

训练阶段分别针对目标检测与语义分割两条分支进行配置：检测模型采用 YOLOv5 微调作为基线，并在 YOLO11 上进行对比实验；分割模型基于 ATTR2UNet，结合 BCE + Dice 损失与强数据增强以提升边界稳定性。两条分支均记录关键训练指标与中间样例，用于后续对比与误差分析。

== Notebook 与训练复现

本项目的训练与可视化笔记本集中在 `notebooks/`：
- `notebooks/attention_r2unet_training.ipynb`（及 v2/v3）：ATTR2UNet 训练与可视化。
- `notebooks/yolo_segmentation_training.ipynb`：YOLO 相关训练与对比实验。

运行方式建议如下：
- 在仓库根目录启动 Jupyter（`jupyter lab` 或 `jupyter notebook`），确保相对路径（如 `data/`, `runs/`, `exports/`）可用。
- 按从上到下顺序执行单元格；单步用 Shift+Enter；需要全量复现时用 “Run All”。若出现环境或缓存问题，先重启 kernel 再全量运行。

模型导出与验证：
- 导出脚本位于 `export_inference.py` 与 `export_attention_r2unet.py`，具体命令见“模型导出与运行（Export & Inference）”小节。
- 导出产物默认输出到 `exports/`，可用 `run_exported_inference.py` 做快速推理检查。


== 数据增强与过拟合对比

在样本量有限且纹理高度重复的 AOI 场景中，模型很容易在训练集上过拟合：训练损失持续下降，但验证集性能停滞甚至回落。我们将训练过程按数据增强强度划分为三种设置，用于观察过拟合风险与泛化能力的变化趋势。

#table(
  columns: (auto, 1fr),
  inset: 6pt,
  stroke: (paint: luma(40%), thickness: 0.8pt),
  align: (left, left),
  [*设置*], [*现象与结论*],
  [无增强], [训练损失快速下降，验证集出现明显性能塌缩；模型倾向记住纹理与噪声，边界与细小缺陷泛化较差。],
  [轻度增强], [训练/验证差距缩小，但对光照、对比度与几何扰动仍较敏感，难以覆盖真实工艺漂移。],
  [强增强], [验证集性能更稳定，过拟合显著缓解；对光照变化、形变与噪声更具鲁棒性，适合小样本或长尾缺陷。],
)

总体而言，当数据规模不足时，强数据增强是抑制过拟合的关键手段；即使在相对较大的数据集上，适度的增强仍能提升模型对域偏移的鲁棒性与可泛化性，因此需要针对 AOI 纹理特性进行精细化调参，而非直接使用默认配置。

== 训练结果

#subpar.grid(
  figure(
    stack(
      spacing: 6pt,
      image("images/no_aug/loss_curve.svg", width: 100%),
      image("images/no_aug/demo.png", width: 100%),
    ),
    caption: [无增强：demo 与 loss curve]
  ), <fig-aug-loss.a>,
  figure(
    stack(
      spacing: 6pt,
      image("images/light_aug/loss_curve.svg", width: 100%),
      image("images/light_aug/demo.png", width: 100%),
    ),
    caption: [轻度增强：demo 与 loss curve]
  ), <fig-aug-loss.b>,
  figure(
    stack(
      spacing: 6pt,
      image("images/with_aug/loss_curve.svg", width: 100%),
      image("images/with_aug/demo.png", width: 100%),
    ),
    caption: [强增强：demo 与 loss curve]
  ), <fig-aug-loss.c>,
  columns: (1fr, 1fr, 1fr),
  caption: [
    #strong[不同增强强度下的训练/验证损失曲线对比。]\
    无增强下训练损失下降更快但验证波动明显；增强后验证曲线更稳定，过拟合得到抑制。
  ],
  label: <fig-aug-loss>,
)

#subpar.grid(
  figure(
    grid(
      "训练集", "测试集",
      image("images/no_aug/samples/train_epoch_0_step0_overlay.png"),
      image("images/no_aug/samples/test_epoch1_step28_overlay.png"),
      image("images/no_aug/samples/train_epoch_8_step250_overlay.png"),
      image("images/no_aug/samples/test_epoch10_step280_overlay.png"),
      image("images/no_aug/samples/train_epoch_32_step900_overlay.png"),
      image("images/no_aug/samples/test_epoch35_step980_overlay.png"),
      columns: (1fr, 1fr)
    ),
    caption: [无增强：预测叠加]
  ), <fig-aug-samples.a>,
  figure(
    grid(
      "训练集", "测试集",
      image("images/light_aug/samples/train_epoch_1_step0_overlay.png"),
      image("images/light_aug/samples/test_epoch1_step28_overlay.png"),
      image("images/light_aug/samples/train_epoch_11_step300_overlay.png"),
      image("images/light_aug/samples/test_epoch10_step280_overlay.png"),
      image("images/light_aug/samples/train_epoch_34_step950_overlay.png"),
      image("images/light_aug/samples/test_epoch35_step980_overlay.png"),
      columns: (1fr, 1fr),
    ),
    caption: [轻度增强：预测叠加]
  ), <fig-aug-samples.b>,
  figure(
    grid(
      "训练集", "测试集",
      image("images/with_aug/samples/train_epoch_1_step0_overlay.png"),
      image("images/with_aug/samples/test_epoch1_step28_overlay.png"),
      image("images/with_aug/samples/train_epoch_11_step300_overlay.png"),
      image("images/with_aug/samples/test_epoch10_step280_overlay.png"),
      image("images/with_aug/samples/train_epoch_34_step950_overlay.png"),
      image("images/with_aug/samples/test_epoch35_step980_overlay.png"),
      columns: (1fr, 1fr)
    ),
    caption: [强增强：预测叠加]
  ), <fig-aug-samples.c>,
  columns: (1fr, 1fr, 1fr),
  caption: [
    #strong[不同增强强度下的可视化示例对比。]\
    强增强条件下边界更稳定、背景误检更少，适合小样本与弱对比缺陷。
  ],
  label: <fig-aug-samples>,
)

== 训练评估

=== 目标检测

YOLOv5 在第 76 个 epoch 达到精确率 0.878、召回率 0.615，mAP\@0.5 为 0.669（mAP\@0.5:0.95 为 0.301）。升级到 YOLO11 后，召回率与 mAP 进一步提升，主要得益于更丰富的主干模块与更积极的数据增强策略。

#figure(
  caption: [检测性能对比（最佳验证轮次）。YOLOv5 指标来自 `runs_aoi_project/yolov5s-aoi-fourcls/results.csv`，YOLO11 指标来自对应的 Kaggle 训练日志。],
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    stroke: (paint: luma(40%), thickness: 0.8pt),
    align: (left, left, left, left, left),
    [*Model*], [*Precision*], [*Recall*], [*mAP\@0.5*], [*mAP\@0.5:0.95*],
    ["YOLOv5s (freeze[0])"], ["0.878"], ["0.615"], ["0.669"], ["0.301"],
    ["YOLO11n (full fine-tune)"], ["0.903"], ["0.672"], ["0.708"], ["0.331"],
  )
) <tbl-yolo>

从定性结果看，YOLOv5 的冻结层减少了灾难性遗忘，但也限制了召回上限；YOLO11 的深度可分离卷积允许更大 batch，提升了对微小墨点的覆盖。需要注意的是，MixUp 系数大于 0.2 时容易引入背景“幻觉”，因此增强策略仍需针对性调参，而非直接采用默认配置。

=== 语义分割

语义分割采用 ATTR2UNet 训练，并在无增强、轻度增强、强增强三种设置下对训练过程进行对比。对应的 loss 曲线与可视化样例见 @fig-aug-loss 和 @fig-aug-samples；数值汇总见表 @tbl-seg-loss。

#figure(
  caption: [分割训练损失对比（E1→E35）。],
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    stroke: (paint: luma(40%), thickness: 0.8pt),
    align: (left, left, left, left),
    [*设置*], [*Train loss (E1→E35)*], [*Val loss (E1→E35)*], [*Gap at E35*],
    [无增强], [0.114 → 0.015], [1.133 → 1.216], [≈ 1.201],
    [轻度增强], [0.171 → 0.025], [0.507 → 0.242], [≈ 0.217],
    [强增强], [0.190 → 0.026], [0.152 → 0.069], [≈ 0.043],
  )
) <tbl-seg-loss>

这些曲线表明：无增强时训练损失快速下降但验证损失持续偏高，泛化差距巨大；轻度增强显著降低验证损失但仍存在一定 gap；强增强同时稳定训练与验证曲线，最终 gap 最小，过拟合得到明显抑制，并在样例可视化中体现为边界更连贯、背景误检更少。

在强增强设置下，ATTR2UNet 的验证损失在前 15 个 epoch 快速收敛，最终保持在较低水平。推理基准结果如下：

#table(
  columns: (auto, auto),
  inset: 6pt,
  stroke: (paint: luma(40%), thickness: 0.8pt),
  align: (left, left),
  [*指标*], [*数值*],
  [推理配置], [单卡 FP16 TorchScript],
  [吞吐量], [22.1 images/s],
  [Dice], [0.9896],
  [IoU], [0.9793],
  [Precision], [0.9898],
  [Recall], [0.9893],
)

尽管整体指标优秀，仍存在边缘侵蚀与二值掩膜过于粗糙的现象，导致边界附近缺陷的 die 归属不稳定；后续可通过多类别掩膜或边界损失进一步改善。

== 集成效果（Integration outcomes）

分割与检测的联动使得 die 级别的规则聚合成为可能，当前系统实现了：

- 对 die-overlap 置信度较低的检测进行标记，便于人工复核；
- 统计每个 die 的缺陷数量与平均置信度，辅助工艺工程师定位热点区域；
- 通过 `run_exported_inference.py` 生成叠加可视化，直观检查缺陷归属与掩膜质量。

#figure(
  image("images/screenshots/demo/demo1.png", width: 100%),
  caption: [
    #strong[分割 + 检测联动可视化示例（导出模型部署）。]\
    该图为导出模型权重集成到 Rust 应用后的可视化结果（https://github.com/jwwang2003/aoi-wafer-stacking）。
  ],
) <fig-integration-demo>

仍需改进的点包括：分割与检测分支置信度未校准，以及任一分支数据更新时可能引入静默回归，因此需要建立联合验证与版本跟踪机制。

=== 模型导出与运行（Export & Inference）

分割模型训练完成后，通过 `export_inference.py` 导出部署权重与 TorchScript；再用 `run_exported_inference.py` 执行推理并生成掩膜/叠加图像，便于下游集成与可视化检查。

```bash
# 导出权重与 TorchScript
python export_inference.py \
  --weights runs/attr2unet/weights/best.pt \
  --out exports/attr2unet

# 运行导出模型推理
python run_exported_inference.py \
  --model exports/attr2unet/attr2unet_ts_fp16.pt \
  --images data/images \
  --out runs/exported_inference
```

上述流程对应的导出产物包括 FP16/FP32 权重与 TorchScript 文件，适合接入 C++/Rust 等部署场景；若需嵌入式/CPU 推理，可在导出阶段增加量化或精简选项。

#pagebreak(weak: true)

= 分析

== 回顾

在小规模标注数据下，数据增强对分割模型的泛化能力有决定性影响。对比无增强与强增强的训练曲线可以看到：不做增强时训练损失迅速下降但验证损失持续偏高，模型容易记住纹理与噪声；而强增强显著收敛了训练/验证差距，使 ATTR2UNet 在小数据集上依旧保持稳定边界与更少误检。

本项目中更有效的增强类型包括：几何增强（翻转、旋转、仿射形变）、光照/色彩增强（亮度、对比度、色调扰动）、模糊与噪声（高斯模糊、噪声注入）。这些增强能够覆盖 AOI 场景中的曝光漂移、工艺纹理变化与成像噪声，从而提升 UNet 系列模型对域偏移的鲁棒性。

对目标检测而言，还需要在训练集中加入 无缺陷的背景图像（不包含标注框）。这类负样本能够显式告知模型“哪些纹理属于正常背景”，有助于压低假阳性并提升在真实产线背景下的稳定性。

此外，训练超参数同样显著影响收敛与过拟合：学习率决定了优化步幅与稳定性，权重衰减控制模型复杂度与泛化，批大小则会影响梯度噪声与正则化强度。合理的组合可以在小数据集上提升稳定性，反之容易出现训练损失下降但验证集退化的过拟合现象。

#grid(
  columns: 3,
  gutter: 10pt,
  box(
    inset: 6pt,
  )[
    #figure(
    grid(
      columns: 1,
      image("images/demo/attention_r2unet_cpu_int8_cpu_image.png"),
      image("images/demo/attention_r2unet_cpu_int8_cpu_mask.png"),
      image("images/demo/attention_r2unet_cpu_int8_cpu_overlay.png"),
    ),
    caption: [
      #strong[分割推理示例（CPU INT8）。]\
      来自 demo 的单图推理结果，用于展示量化后的边界质量与背景抑制效果。
    ],
  )],
  box(
    inset: 6pt,
  )[
    #figure(
    grid(
      columns: 1,
      image("images/demo/attention_r2unet_cpu_fp32_cpu_image.png"),
      image("images/demo/attention_r2unet_cpu_fp32_cpu_mask.png"),
      image("images/demo/attention_r2unet_cpu_fp32_cpu_overlay.png"),
    ),
    caption: [
      #strong[分割推理示例（CPU FP32）。]\
      来自 demo 的单图推理结果，用于展示边界质量与背景抑制效果。
    ],
  )],
  box(
    inset: 6pt,
  )[
    #figure(
    grid(
      columns: 1,
      image("images/demo/attention_r2unet_gpu_fp16_cuda0_image.png"),
      image("images/demo/attention_r2unet_gpu_fp16_cuda0_mask.png"),
      image("images/demo/attention_r2unet_gpu_fp16_cuda0_overlay.png"),
    ),
    caption: [
      #strong[分割推理示例（GPU FP16）。]\
      来自 demo 的单图推理结果，用于展示半精度推理的边界质量与背景抑制效果。
    ],
  )],
)

== 最后推理结果

#table(
  columns: (auto, 1fr),
  inset: 6pt,
  stroke: (paint: luma(40%), thickness: 0.8pt),
  align: (left, left),
  [*推理精度格式*], [*结论与特性*],
  [GPU FP16], [速度最快，吞吐高，适合在线部署；但对显存与硬件依赖更强。],
  [CPU FP32], [精度与稳定性最保守，结果更接近训练分布；速度中等，适合离线或验证。],
  [CPU INT8], [速度与资源占用最优，但存在量化误差风险；适合对吞吐敏感的轻量部署。],
)

总体比较：FP16 在 GPU 上提供最佳吞吐，但数值精度略低；CPU FP32 最稳定、可复现实验结果；INT8 以牺牲部分数值精度换取更低延迟与更小模型体积，适合资源受限场景。实际选择需在吞吐、精度与部署成本之间折中。

== 数据层面的不足

- 边缘缺陷样本稀缺。 崩边类样本占比低，可通过主动学习优先采集芯片边缘高不确定性样本，以提升关键区域召回率。
- 标注漂移。 *Acceptable* 框有时覆盖真实缺陷，导致模型偏向“无缺陷”判断。引入多边形标注或“可信掩膜”通道可减少监督冲突。
- 跨批次差异。 不同 collection 在曝光和晶圆 pitch 上差异明显，可通过分批次色彩归一化或对抗式域对齐缓解分布偏移。

== 模型与流程的不足

- YOLO 召回上限。 Anchor 机制缺少结构先验，小目标墨点易被忽略。可在分割后的单 die 上裁剪并运行轻量检测器以提高分辨率。
- 置信度未校准。 检测与分割均输出未校准概率，可采用温度缩放 + 保序校准来更真实反映不确定性。
- 掩膜二值化限制。 ATTR2UNet 仅输出单一 die 类别，区域判断依赖启发式。训练多类别掩膜（中心/布线/保护环）并加入边界损失可直接编码区域信息。
- 训练异步。 分割与检测独立迭代，数据更新可能悄然破坏 die 归属逻辑，应引入联合验证或全景式训练以避免回归。
- 误差传播。 分割缺失会将该区域检测全部判为 off-die。加入“未知 die”状态并结合几何网格检测（如霍夫拟合）可提升鲁棒性。

== 改进方向

- 自动化验证套件。 为每个验证 batch 生成检测/分割叠加图的 Typst 附录，尽早发现标注漂移。
- 边缘导向增强。 用可控光照合成 die 边缘并粘贴真值芯片，替代全局 MixUp，针对最难类别定向增强。
- 多尺度推理。 以 640 与 960 双尺度运行 YOLO11 并融合结果，分割掩膜可作为空间先验降低融合复杂度。
- 芯片内坐标回归。 训练小型回归器融合检测特征与 die 掩膜裁剪，输出连续 (u, v) 坐标，将“区域判断”纳入可学习路径。

= 结论

YOLOv5 提供了稳定的基线，YOLO11 在召回上取得提升，Attention R2U-Net 输出了高质量 die 掩膜；但更关键的收获是对系统性薄弱环节的识别。后续通过解决数据不平衡、置信度校准以及分割与检测的紧耦合问题，可将本项目从验证性方案推进为可审计、可量产评审的 AOI 检测流水线。

#pagebreak()

#bibliography("references.bib")
