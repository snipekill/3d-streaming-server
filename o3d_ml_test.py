import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

cfg_file = "ml3d/configs/pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/path/to/your/dataset"
dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# ...
# # run inference on a single example.
# result = pipeline.run_inference(data)