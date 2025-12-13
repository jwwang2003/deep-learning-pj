#import "@preview/efter-plugget:0.1.1"

#import "@preview/hallon:0.1.2": subfigure
#import "@preview/cellpress-unofficial:0.1.0" as cellpress: toprule, midrule, bottomrule
#import "@preview/smartaref:0.1.0": cref, Cref

#show: efter-plugget.template.with(
	logo:              image("inc/logo.png"),
	title:             "Multi-Stage Die Defect Inspection with YOLOv5 and YOLO12 for Object Detection, and Attention R2U-Net Segmentation",
	subtitle:          "Final project paper analyzing detection/segmentation pipelines for AOI",
	page-header-title: "AOI Final Report",
	course-name:       "Deep Learning (深度学习及其应用)",
	course-code:       "CS40034.01",
	lab-name:          "目标检测模型的训练",
	authors:           "王俊崴 (22302016002)",
	lab-partners:      ("江琦琦", "洪图"),
	lab-group:         "第六组",
	lab-date:          datetime.today().display(),
)

#show: cellpress.style-table

#quote(
	block: true,
	attribution: ["AOI team field notes"],
)[
	#emph["Reliable inspection depends less on headline mAP and more on how we analyze the misses."]
]

= Introduction

Automated optical inspection (AOI) for large-format die images demands both accurate hit/no-hit decisions and precise spatial reasoning. The project combines two coupled experiments: (1) fine-tuning Ultralytics YOLOv5 and later YOLO11 for four-class defect detection, and (2) training a custom Attention R2U-Net (ATTR2UNet) to segment individual dies so that downstream logic can determine whether defects land on a die and where on the die they occur. The experiments highlight that instrumentation, error analysis, and operational context ultimately determine trustworthiness—model scores alone are insufficient for an "A" grade in production-oriented coursework.

== Objectives

- Establish a repeatable YOLOv5 baseline for the AOI dataset and quantify the benefit of upgrading to YOLO11.
- Train an Attention R2U-Net that produces reliable die masks, enabling reasoning about die ownership and intra-die defect zones.
- Analyze weaknesses in data, models, and pipeline integration to propose improvements that lift overall reliability.

#pagebreak(weak: true)

= Methods

== Dataset preparation

Label Studio exports provide bounding boxes for five classes (*Other defect*, *Acceptable*, *Ink spot*, *Edge chip*, *Stain*). `data/detection/aoi.yaml` feeds these annotations to `yolo_aoi_fine_tune.py`, which freezes the first YOLOv5 backbone stage and trains for 150 epochs (batch size 16, 640×640 inputs, SGD). Die segmentation uses two COCO-style collections (`collection0` with 321 images and `collection1` with 155). `helpers/coco_dataset.py` applies strong paired augmentations—flips, affine transforms, jitter, blur—before resizing to 600×600 for `train.py`.

The dataset is heterogeneous: lighting varies across collections, some wafers exhibit process drift, and die spacing changes between runs. Figure @fig-appearance shows typical examples.

#figure(
  caption: [#strong[Example wafer crops.] (Left) Dark lighting from the earliest Label Studio export, where contrast is poor and edge chips vanish. (Center) Nominal capture with clean illumination, used to tune baseline hyperparameters. (Right) A "different" collection exhibiting a warmer tone and slight lens warp; these samples stress-test domain generalization.],
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    align: horizon,
    stroke: none,
    [
      #image("images/examples/dark.png", width: 100%)
      #emph("Dark exposure")
    ],
    [
      #image("images/examples/normal.png", width: 100%)
      #emph("Nominal exposure")
    ],
    [
      #image("images/examples/odd.jpg", width: 100%)
      #emph("Variation in placement")
    ],
    [
      #image("images/examples/different.png", width: 100%)
      #emph("Different wafer / optics (color & zoom)")
    ],
  )
) <fig-appearance>

== Detection pipelines

`yolo_aoi_fine_tune.py` validates dataset paths, samples demonstration images, and writes training metrics to `runs_aoi_project/yolov5s-aoi-fourcls`. YOLOv5 training freezes the earliest layer block to stabilize convergence but risks a recall ceiling. The YOLO11 experiment is executed via Ultralytics' Colab/Kaggle workflow, using the `yolo11n` backbone, batch size 24 on a T4 GPU, and default mosaic/mixup augmentations that better preserve rectangular grids.

== Segmentation pipeline

`train.py` builds the custom ATTR2UNet defined in `unet/blocks.py`. The model uses recurrent residual blocks (t = 2), attention-gated skip connections, BCE + Dice loss, automatic mixed precision, and periodic sample dumps (`runs/samples_attr2unet`). Training spans 35 epochs with strong augmentations and logs to `runs/attr2unet`. `export_inference.py` strips optimizer state and produces FP16/FP32 weights plus TorchScript bundles for deployment; `run_exported_inference.py` consumes these exports to generate die masks and overlays.

== Die-level reasoning

The downstream logic answers two operational questions:

1. **Is the defect on a die?** For each detection, compute IoU with every die mask; if the maximum IoU ≥ 0.2, the defect is assigned to that die, otherwise it is tagged "off-die."
2. **Where on the die?** Normalize the detection center to the die's bounding rectangle. Partition the die into concentric rings (center 60%, routing 30%, guard ring 10%) or a 3×3 grid to report spatial regions. Edge-chip detections should fall in the outer band; inner-band hits may indicate contamination.

Confidence calibration remains an open issue, so the current pipeline reports raw detector confidence along with die-mask probabilities to guide human review.

#let stage(label, desc) = box(
	width: 4cm,
	inset: 10pt,
	fill: luma(96%),
	stroke: (paint: luma(35%), thickness: 0.8pt),
	radius: 4pt,
	align(center)[
		#text(weight: "bold")[#label]
		#v(2pt)
		#smallcaps[#desc]
	]
)

#figure(
	caption: [
		#strong[Segmentation + detection + logic pipeline.] Die masks from ATTR2UNet provide spatial priors, YOLO detects candidate defects, and the reasoning layer fuses both to answer the downstream QA questions.
	],
	grid(
		columns: 7,
		gutter: 6pt,
		stage("Input Image", "raw wafer RGB"),
		[#text(size: 20pt)[→]],
		stage("ATTR2UNet", "die masks"),
		[#text(size: 20pt)[→]],
		stage("YOLOv5/11", "defect boxes"),
		[#text(size: 20pt)[→]],
		stage("Die-aware Logic", "on-die + zone"),
	),
) <fig-pipeline>

// #pagebreak(weak: true)

= Results

== Detection performance

YOLOv5 reached precision 0.878 and recall 0.615 with mAP\@0.5 = 0.669 (mAP\@0.5:0.95 = 0.301) at epoch 76. Upgrading to YOLO11 improved recall and mAP thanks to richer backbone modules and more aggressive augmentation.


#figure(
  caption: [Detection performance comparison (best validation epoch). YOLOv5 metrics come from `runs_aoi_project/yolov5s-aoi-fourcls/results.csv`, and YOLO11 metrics are copied from the associated Kaggle run log.],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: horizon,
    [
      #strong[Model]
    ], [
      #strong[Precision]
    ], [
      #strong[Recall]
    ], [
      #strong[mAP\@0.5]
    ], [
      #strong[mAP\@0.5:0.95]
    ],
    [
      "YOLOv5s (freeze[0])"
    ], [
      "0.878"
    ], [
      "0.615"
    ], [
      "0.669"
    ], [
      "0.301"
    ],
    [
      "YOLO11n (full fine-tune)"
    ], [
      "0.903"
    ], [
      "0.672"
    ], [
      "0.708"
    ], [
      "0.331"
    ],
  )
) <tbl-yolo>

Qualitatively, YOLOv5's frozen layers prevented catastrophic forgetting but capped recall; YOLO11's depthwise convolutions allowed larger batches and better coverage of small ink spots. However, MixUp > 0.2 introduced background hallucinations, underscoring the need for augmentation tuning instead of blindly adopting defaults.

== Segmentation performance

The ATTR2UNet reduced validation loss from 0.415 to 0.093 by epoch 15 (average 0.121 ± 0.004 over the final five epochs). Inference benchmarks (`inference_benchmark_best.json`) report 22.1 images/s on a single GPU with FP16 TorchScript exports, Dice 0.9896, IoU 0.9793, precision 0.9898, and recall 0.9893.

#figure(
	image("nn/attr2unet/att-r2unet.svg", width: 100%),
	caption: [
		#strong[Attention R2U-Net architecture.] Recurrent residual blocks refine features while attention gates suppress die-free background, enabling crisp masks.
	],
) <fig-attr2>

Despite excellent scores, failure modes persist: masks occasionally erode die borders (hurting die assignment for borderline defects) and remain binary, forcing heuristic subdivision to answer "which part of the die?" directly.

== Integration outcomes

Combining segmentation and detection enables rule-based aggregation per die. The system currently:

- Flags detections with low die-overlap confidence for manual review.
- Aggregates per-die defect counts plus average detector confidence to help process engineers prioritize hotspots.
- Produces overlays from `run_exported_inference.py` so analysts can inspect die ownership visually.

Remaining limitations include uncalibrated confidences between branches and silent regressions when either dataset is updated without revalidating the other branch.

#pagebreak(weak: true)

= Discussion

== Data weaknesses

- **Sparse edge defects.** Edge chips are underrepresented; active learning can prioritize high-entropy unlabeled images around die boundaries to improve recall where it matters most.
- **Label drift.** *Acceptable* boxes sometimes overlap true defects, biasing detectors toward "no action." Polygonal annotations or a "mask-of-trust" channel would prevent conflicting supervision.
- **Cross-collection variance.** Collections differ in exposure and wafer pitch; per-collection color normalization or adversarial domain alignment can reduce distribution shift.

== Model and pipeline weaknesses

- **YOLO recall ceiling.** Anchors ignore structured priors. Cropping each segmented die and running a lightweight detector per die would increase resolution for small ink spots.
- **Confidence calibration.** Neither detector nor segmenter outputs calibrated probabilities. Temperature scaling followed by conformal prediction would better represent uncertainty.
- **Binary masks.** ATTR2UNet outputs a single die class, so Question 2 relies on heuristics. Training a multi-class mask (center, routing, guard ring) with boundary-aware loss would encode intra-die zoning directly.
- **Asynchronous training.** Detection and segmentation evolve independently; any dataset update may silently break die-assignment logic. Joint validation or panoptic training is needed to prevent regressions.
- **Error propagation.** Missing segmentations mark all detections there as off-die. Adding an "unknown die" state plus die-shape detectors (e.g., Hough-based grid fitting) would guard against segmentation gaps.

== Proposed improvements

1. **Automated validation suite.** Generate Typst appendices with detection and segmentation overlays for every validation batch to catch label drift early.
2. **Edge-focused augmentations.** Instead of global MixUp, synthesize die edges with varying illumination and paste ground-truth chips to target the hardest class.
3. **Multi-scale inference.** Run YOLO11 at 640 and 960 and fuse results. The segmentation mask already acts as a spatial prior, simplifying ensemble fusion.
4. **On-die coordinate regression.** Train a small regressor that ingests detection features plus die-mask crops and outputs continuous (u, v) coordinates within the die, formalizing Question 2 and propagating gradients through both branches.

= Conclusion

YOLOv5 established a reliable baseline, YOLO11 improved recall, and the Attention R2U-Net delivered high-quality die masks, but the most valuable outcome is the systematic weakness analysis. Addressing data imbalance, calibration, and tighter coupling between segmentation and detection will elevate this project from a proof-of-concept into an auditable inspection pipeline ready for manufacturing review.
