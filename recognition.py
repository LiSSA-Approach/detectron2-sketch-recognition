from ctypes import Union
from typing import Dict
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

classes = ['ClassNode', 'Qualifier', 'NAryAssociationDiamond', 'Package', 'Comment', 'Label', 'Aggregation', 'Composition', 'Extension', 'Dependency', 'Realization', 'CommentConnection', 'AssociationUnidirectional', 'AssociationBidirectional']

class Recognition:
   
    def __init__(self) -> None:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(".", "model", "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

        self._predictor = DefaultPredictor(cfg)


    def interprete(self, data: bytes):#  -> Union[str, dict]:
        try:
            npdata = np.fromstring(data, np.uint8)
            image = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
            outputs = self._predictor(image)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        except Exception as error:
            return f"Could not predict: {error}"
        
        instances = outputs["instances"]
        boxes = instances.pred_boxes
        scores = instances.scores
        prediction_classes = instances.pred_classes

        predictions = []
        idx = 0
        for box in boxes:
            entry = {
                "box": [tensor.item() for tensor in box],
                "confidence": scores[idx].item(),
                "class": classes[prediction_classes[idx]]
            }
            predictions.append(entry)
            idx += 1

        return predictions
