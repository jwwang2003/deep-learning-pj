# deep-learning-pj

Completed tasks:
- image \(DIE\) segmentation
- object \(defect\) detection

## Setting up

Create a new Conda environment that uses `Python 3.10`:

```
# Replace 'cv' with anything that you prefer to name the environment
conda create --name cv python=3.10
 pip install -r requirements.txt
 ```
 
 Initialize the YOLOv5 submodule:
 
 ```
 git submodule update --init --recursive
 ```

If you run into the following error when running the Python scripts:

```
ImportError:
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.1.2 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
```

Try to downgrade Numpy: `pip install "numpy<2"`.

You might also see this error about OpenCV, we do not use it in this project
so you can ignore the warning.

```
(yolov5) PS C:\Users\wjw_0\Downloads\deep-learning-pj\deep-learning-pj\yolov5> pip install --upgrade --force-reinstall "numpy<2.0"
Looking in indexes: https://pypi.org/simple, https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
Collecting numpy<2.0
Using cached https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/b5/42/054082bd8220bbf6f297f982f0a8f5479fcbc55c8b511d928df07b965869/numpy-1.26.4-cp39-cp39-win_amd64.whl (15.8 MB)
Installing collected packages: numpy
Attempting uninstall: numpy
Found existing installation: numpy 2.0.2
Uninstalling numpy-2.0.2:
Successfully uninstalled numpy-2.0.2
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
Successfully installed numpy-1.26.4
```

## Dataset

In our project, we choose to use Label Studio as our main tool for hand labelling
our custom dataset.

Within the root of the project, run the following command to instantly start labelling:
```
docker run -it -p 8080:8080 -v ${PWD}/mydata:/label-studio/data heartexlabs/label-studio:latest label-studio --log-level DEBUG
```

## Training & running inference

Run the following commands in the root project folder.

### Training
From the repo root (after activating the env above):
```
python yolo_aoi_fine_tune.py
```
This launches YOLOv5 fine-tuning with outputs under `runs_project1/`.

### Inference on the test split
Uses the freshest `best.pt` checkpoint saved under `runs_project1/*/weights/`.
```
python yolo_aoi_test_demo.py
```
Results land in `runs_project1/test_demos/<run-name>-<split>/`.

python train.py --data-root data\segmentation\project-1-at-2025-12-11-06-33-9fc6a7a1 \
--coco-json data\segmentation\project-1-at-2025-12-11-06-33-9fc6a7a1\result.json \
--images-dir data\segmentation\project-1-at-2025-12-11-06-33-9fc6a7a1\images \
--masks-dir data\segmentation\project-1-at-2025-12-11-06-33-9fc6a7a1\masks
--model attr2unet --batch-size 4 --num-workers 4

(base) PS B:\projects\deep-learning\deep-learning-pj> python train.py `                
>>   --data-root "data\segmentation\project-1-at-2025-12-11-06-33-9fc6a7a1" `
>>   --coco-json "data\segmentation\project-1-at-2025-12-11-06-33-9fc6a7a1\result.json" `
>>   --images-dir "data\segmentation\project-1-at-2025-12-11-06-33-9fc6a7a1\images" `
>>   --masks-dir "data\segmentation\project-1-at-2025-12-11-06-33-9fc6a7a1\masks" `  
>>   --model attr2unet --batch-size 3 --sample-every 50 --sample-count 4 --sample-dir runs/samples_attr2

python -m helpers.s3_fetch data/segmentation/project-1-at-2025-12-11-06-33-9fc6a7a1/result.json --out data/segmentation/project-1-at-2025-12-11-06-33-9fc6a7a1/images --env-file .env

## Exporting inference-ready weights

The training checkpoints (e.g. `runs/attr2unet/best.pt`) store optimizer/state that is
useful for resuming but bloats deployment artifacts. Use `export_inference.py` to produce
lean weights plus TorchScript bundles that are optimized for GPU and CPU inference:

```
python export_inference.py \
  --checkpoint runs/attr2unet/best.pt \
  --config runs/attr2unet/config.json \
  --image-size 600 \
  --output-dir exports/attr2unet \
  --targets weights gpu-script cpu-script cpu-int8 vanilla
```

Artifacts created:
- `inference_fp32.pt`: weights only (float32).
- `inference_fp16.pt`: FP16 weights for GPU deployment (~2× smaller).
- `model_gpu_fp16.ts`: TorchScript traced in half precision (requires CUDA to export).
- `model_cpu_fp32.ts`: TorchScript traced on CPU with `torch.jit.optimize_for_inference`.
- `model_cpu_int8.ts`: optional FX-quantized script for CPU (falls back gracefully if quantization fails).
- `checkpoint_vanilla.pt`: optional copy of the original training checkpoint (include `vanilla` in `--targets`).

Load the weight-only files with the usual model builders (`AttentionR2UNet`, etc.).
TorchScript bundles can be loaded with `torch.jit.load` directly in inference services.

### Batch inference on arbitrary images

After exporting, you can run the trained model on any set of images without COCO
annotations via `run_exported_inference.py`:

```
python run_exported_inference.py \
  --weights exports/attr2unet/model_cpu_fp32.ts \
  --images /path/to/img1.png /path/to/img2.jpg \
  --output-dir runs/custom_inference \
  --threshold 0.45 \
  --save-overlay
```

`--weights` accepts either the TorchScript bundles (`.ts`) or the weight-only
snapshots (`inference_fp32.pt`, etc.). Masks are saved as `<stem>_mask.png` and
optional overlays highlight predictions in red for quick inspection.

### YOLOv5 weight extraction

Detection runs live under `runs_aoi_project/<run-name>`. To strip a YOLOv5
checkpoint down to CPU FP32 and GPU FP16 inference weights, run:

```
python -m helpers.export_yolov5_weights \
  --checkpoint runs_aoi_project/yolov5s-aoi-fourcls/weights/best.pt \
  --output-dir exports/yolov5s-aoi-fourcls
```

The helper writes `<run-name>_cpu_fp32.pt` and `<run-name>_gpu_fp16.pt` inside
the chosen output directory—both keep the YOLOv5 model object but drop optimizer
state for light deployment on CPU or CUDA hosts.

### TorchScript smoke test

After exporting YOLOv5 TorchScript bundles (via `yolov5/export.py`), run:

```
python -m helpers.test_yolov5_torchscript \
  --weights exports/yolov5s-aoi-fourcls/yolov5s-aoi-fourcls.ts \
  --images data/detection/aoi_demo/*.jpg \
  --output-dir runs/yolov5_ts_eval
```

The script prints detections for each image and (optionally) writes annotated
frames, making it easy to confirm that the `.ts` artifact behaves as expected
before wiring it into production services.

## Downloading the dataset \(\)

Populate `.env` file and run the S3 helper script to download the images inside of the bucket.

### Collection 0 \(stored privately on Amazon S3\)

```
python -m helpers.s3_data data/segmentation/collection0/result.json --out data/segmentation/collection0/images --env-file .env
```

### Collection 1 \(stored privately on Amazon S3\)

The S3 python script can also renmaed the links in case the storage location has changed:

```
python -m helpers.s3_data data/segmentation/collection1/result.json \
  --rename --bucket sichain-aoi-datasets --prefix "jqq" --download
```

```
python -m helpers.s3_data data/segmentation/collection2/result.json \
  --bucket sichain-aoi-datasets --download
```

### Collection 2 \(stored privately on Amazon S3\)


### Generate some preview masks and overlays to ensure the data is OK

```
python -m helpers.generate_masks data/segmentation/collection0/result.json --limit 10 --overlay
python -m helpers.generate_masks data/segmentation/collection1/result.json --limit 10 --overlay
```

```
label-studio-converter import coco -i data/segmentation/collection1/result.json -o data/segmentation/collection1/label_annotation.json
```

```
python test.py data/segmentation/collection3/la
bel_annotation.json -o  data/segmentation/collection3/label.json
```
