import json
import requests
# from constants import *
from typing import List, Optional
import os
from enum import IntEnum
import yaml
import json
import subprocess


class NLPTask(IntEnum):
    TEXT_CLASSIFICATION = 0
    NAMED_ENTITY_RECOGNITION = 1
    PROMPTING = 2
    CHATBOT = 3
    UNSUPERVISED = 4
    OBJECT_DETECTION = 5

class ModelType(IntEnum):
    NO_LABEL_TEXT_CLASSIFICATION = 0
    FEW_SHOT_TEXT_CLASSIFICATION = 1
    NAIVE_BAYES_TEXT_CLASSIFICATION = 2
    SETFIT_TEXT_CLASSIFICATION = 3
    NOT_ALL_TEXT_CLASSIFICATION = 4
    FEW_SHOT_NAMED_ENTITY_RECOGNITION = 5
    EXAMPLE_BASED_NAMED_ENTITY_RECOGNITION = 6
    GPT_FOR_PROMPTING = 7
    PROMPT_NAMED_ENTITY_RECOGNITION = 8
    PROMPTING_WITH_FEEDBACK_PROMPT_ENGINEERED = 9
    DUMMY = 10
    GPT_FINETUNING = 11
    RAG_UNSUPERVISED = 12
    ZEROSHOT_GPT4 = 13
    ZEROSHOT_CLAUDE = 14
    ZEROSHOT_LLAMA3 = 15
    ZEROSHOT_MISTRAL = 16
    ZEROSHOT_GPT4MINI = 17
    ZEROSHOT_GEMINI = 18
    FASTER_RCNN = 19
    YOLO = 20
    GROUNDING_DINO = 21
    
class ZeroShotModelType(IntEnum):
    ZERO_SHOT_GPT4 = 0
    ZERO_SHOT_CLAUDE = 1
    ZERO_SHOT_LLAMA3 = 2
    ZERO_SHOT_MISTRAL = 3
    ZERO_SHOT_GPT4_MINI = 4
    ZERO_SHOT_GEMINI = 5

class EvaluationMetric(IntEnum):
    COSINE_SIM = 0
    BERT_SCORE = 1
    ROUGE_L_F1 = 2
    FAITHFULNESS = 3
    ANSWER_RELEVANCE = 4
    ANOTE_MISLABEL_SCORE = 5
    CONFUSION_MATRIX = 6
    CLASSIFICATION_REPORT = 7
    PRECISION = 8
    RECALL = 9
    F1 = 10
    IOU = 11
    SUPPORT = 12
    MAP = 13
    TP = 14
    FP = 16
    FN = 17

class Anote:
    def __init__(self, api_key):
        self.api_key = api_key
        # Initialize any required configurations

    # def upload(self, dataset_name: str, data_path: str, split: str):
    #     """
    #     Uploads a dataset and registers it for training or evaluation.
        
    #     Args:
    #         dataset_name (str): A unique name to register the dataset.
    #         data_path (str): Path to a .json file containing image paths, labels, and bounding boxes.
    #         split (str): One of 'train', 'validation', 'test'.
            
    #     Returns:
    #         dict: Should return a dictionary with:
    #             - "status": str - Confirmation message
    #             - "dataset_id": str - Unique identifier for the uploaded dataset
    #     """
    #     # Logic to upload dataset would go here
    #     print(f"Uploading dataset {dataset_name} from {data_path} for {split} split")
        
    #     # Should return confirmation and dataset ID
    #     return {"status": "Dataset uploaded successfully", "dataset_id": "dataset_123"}

def upload(self, dataset_name: str, data_path: str, split: str, task_type: int = NLPTask.TEXT_CLASSIFICATION, model_type: str = None):
    """
    Upload a dataset for NLP or CV tasks with proper format validation.
    For CV, validates YOLO, Faster R-CNN, and Grounding DINO formats.
    """
    if task_type == NLPTask.OBJECT_DETECTION:
        assert model_type in ["yolo", "faster_rcnn", "grounding_dino"], "model_type must be one of: yolo, faster_rcnn, grounding_dino"

        print(f"[UPLOAD] Validating object detection dataset for {model_type}")

        if model_type == "yolo":
            # Expects: {train, val, test} folders + dataset.yaml + .txt labels per image
            assert os.path.isdir(data_path), f"{data_path} is not a valid directory"
            yaml_path = os.path.join(data_path, "dataset.yaml")
            assert os.path.isfile(yaml_path), "yolo dataset must contain dataset.yaml"

            for split_name in ["train", "val", "test"]:
                split_dir = os.path.join(data_path, split_name)
                assert os.path.isdir(split_dir), f"Missing {split_name} folder in YOLO dataset"
                images_dir = os.path.join(split_dir, "images")
                labels_dir = os.path.join(split_dir, "labels")
                assert os.path.isdir(images_dir), f"Missing images folder in {split_name}"
                assert os.path.isdir(labels_dir), f"Missing labels folder in {split_name}"
                for img_file in os.listdir(images_dir):
                    if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                        label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
                        assert os.path.isfile(label_file), f"Missing label file for {img_file}"

        elif model_type == "faster_rcnn":
            # Expect: annotations in COCO JSON format + images folder
            assert os.path.isdir(data_path), f"{data_path} is not a valid directory"
            ann_file = os.path.join(data_path, f"annotations_{split}.json")
            assert os.path.isfile(ann_file), f"Missing COCO annotations JSON for {split}"
            images_dir = os.path.join(data_path, split)
            assert os.path.isdir(images_dir), f"Missing images folder for {split} split"
            with open(ann_file, "r") as f:
                ann_data = json.load(f)
            assert "images" in ann_data and "annotations" in ann_data, \
                "COCO JSON must have 'images' and 'annotations' keys"

        elif model_type == "grounding_dino":
            # Expect: annotation text files with grounding captions + image folder
            assert os.path.isdir(data_path), f"{data_path} is not a valid directory"
            split_dir = os.path.join(data_path, split)
            assert os.path.isdir(split_dir), f"Missing folder for split {split}"
            images_dir = os.path.join(split_dir, "images")
            captions_dir = os.path.join(split_dir, "captions")
            assert os.path.isdir(images_dir), f"Missing images folder in {split}"
            assert os.path.isdir(captions_dir), f"Missing captions folder in {split}"
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    caption_file = os.path.join(captions_dir, os.path.splitext(img_file)[0] + ".txt")
                    assert os.path.isfile(caption_file), f"Missing caption file for {img_file}"

        print(f"[UPLOAD] {model_type} dataset for {split} split is in correct format.")
        return {"status": "Dataset uploaded successfully", "dataset_id": f"{dataset_name}_{split}"}

    else:
        # NLP upload stuff
        print(f"[UPLOAD] NLP dataset {dataset_name} ({split}) from {data_path}")
        return {"status": "Dataset uploaded successfully", "dataset_id": f"{dataset_name}_{split}"}


def train(self, task_type: int, model_type: str, train_dataset: str, validation_dataset: str = None):
    """
    Trains an object detection model using the provided dataset.
    
    Args:
        task_type (int): Type of task. Use 5 for object detection.
        model_type (str): Type of model ('faster_rcnn', 'yolo', 'grounding_dino').
        train_dataset (str): Path to training dataset JSON.
        validation_dataset (str, optional): Path to validation dataset JSON.
        
    Returns:
        dict: Should return a dictionary with:
            - "model_id": str - A unique identifier for the trained model
            - "status": str - Training status message
            - "training_job_id": str - ID to track training progress
    """
    assert task_type == 5  , "Object detection is supported by task_type=5."
    assert model_type in ["faster_rcnn", "yolo", "grounding_dino"], "model_type must be one of: faster_rcnn, yolo, grounding_dino."
    
    print(f"[TRAIN] Starting training for {model_type} with dataset {train_dataset}")

    # CHANGE THESE PATHS TO ACTUAL TRAINING SCRIPTS
    script_paths = {
        "yolo": "OpenAnote/materials/research/researchcode/Benchmarking-ObjectDetection/FIAR1M/fine_tuned_yolov11/[Fine_Tuned]_FAIR1M_YOLOv8.ipynb",
        "faster_rcnn": "./faster_rcnn/train.py", # fake path
        "grounding_dino": "./grounding_dino/train.py" # fake path
    }

    assert model_type in script_paths, f"Unsupported model_type '{model_type}'. Choose from {list(script_paths.keys())}."
    script_path = script_paths[model_type]

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Training script not found at: {script_path}")

    print(f"[TRAIN] Starting {model_type} training...")
    cmd = [
        "python", script_path,
        "train", train_dataset
    ]

    if validation_dataset:
        cmd += ["val", validation_dataset]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training script failed: {e}")
        return {"status": "training_failed", "error": str(e)}

    # Generate unique IDs for model and training job
    model_id = f"{model_type}_{uuid.uuid4().hex[:8]}"
    training_job_id = f"job_{uuid.uuid4().hex[:8]}"

    print(f"[TRAIN] {model_type} training started successfully.")
    return {
        "model_id": model_id,
        "status": "training_started",
        "training_job_id": training_job_id
    }

    # if model_type == 'yolo':
    #     print(f"Training yolo model on {train_dataset}")
    # elif model_type == 'faster_rcnn':
    #     print(f"Training Faster R-CNN model on {train_dataset}")
    # elif model_type == 'grounding_dino':
    #     print(f"Training Grounding DINO model on {train_dataset}")

    # # Should return model ID and training status
    # return {"model_id": "model_456", "status": "training_started", "training_job_id": "job_789"}

def predict(self, model_type: str, test_data: str, labels: List[str], model_id: str = None, confidence_threshold: float = 0.5):
    """
    Runs inference on the provided test images using a trained model.
    
    Args:
        model_type (str): Model to use ('faster_rcnn', 'yolo', 'grounding_dino').
        test_data (str): Path to image files or JSON list.
        labels (List[str]): List of possible class labels.
        model_id (str, optional): ID of a pre-trained model.
        confidence_threshold (float, optional): Filter out low-confidence predictions.
        
    Returns:
        List[dict]: Should return a list of predictions, each containing:
            - "image_id": int/str - Unique identifier for the image
            - "boxes": List[List[float]] - Bounding boxes in [x1, y1, x2, y2] format
            - "labels": List[str] - Predicted class names
            - "confidence": List[float] - Confidence scores for each prediction
    """
    print(f"Running {model_type} predictions on {test_data} with model {model_id}")
    
    # Should return list of prediction dictionaries
    return [
        {
            "image_id": 0,
            "boxes": [[10, 10, 50, 50], [60, 60, 100, 100]],
            "labels": ["A5", "A9"],
            "confidence": [0.8, 0.7]
        }
    ]

def evaluate(self, ground_truths: str, predictions: str):
    """
    Evaluates the predictions against ground truth annotations.
    
    Args:
        ground_truths (str): Path to the ground truth JSON file.
        predictions (str): Path to the predictions JSON file.
        
    Returns:
        dict: Should return evaluation results containing:
            - "precision": float - Overall precision score
            - "recall": float - Overall recall score  
            - "f1_score": float - F1 score
            - "accuracy": float - Accuracy score
            - "mAP": float - Mean Average Precision
            - "mIoU": float - Mean Intersection over Union
            - "confusion_matrix_path": str - Path to generated confusion matrix image
            - "metrics_csv_path": str - Path to detailed metrics CSV file
            - "per_class_metrics": dict - Per-class precision, recall, F1 scores
    """
    print(f"Evaluating predictions from {predictions} against ground truth {ground_truths}")
    
    # Should return comprehensive evaluation metrics
    return {
        "precision": 0.85,
        "recall": 0.78,
        "f1_score": 0.81,
        "accuracy": 0.83,
        "mAP": 0.72,
        "mIoU": 0.68,
        "confusion_matrix_path": "confusion_matrix.png",
        "metrics_csv_path": "metrics.csv",
        "per_class_metrics": {
            "A5": {"precision": 0.82, "recall": 0.79, "f1": 0.80},
            "A9": {"precision": 0.88, "recall": 0.77, "f1": 0.82}
        }
    }

def checkStatus(self, predict_report_id=None, model_id=None):
    """
    Check the status of a prediction or training process.

    Args:
        predict_report_id (str, optional): The ID of the prediction report.
        model_id (str, optional): The ID of the model.

    Returns:
        dict: Should return status information containing:
            - "isComplete": bool - Whether the process is finished
            - "status": str - Current status description
            - "progress": float - Completion percentage (0-100)
            - "estimated_time_remaining": int - Seconds remaining (if available)
            - "error_message": str - Error details if failed
    """
    print(f"Checking status for model_id: {model_id}, predict_report_id: {predict_report_id}")
    
    # Should return current status information
    return {
        "isComplete": True,
        "status": "completed",
        "progress": 100.0,
        "estimated_time_remaining": 0,
        "error_message": None
    }

def predictAll(self, report_name, model_types, model_id, dataset_id, actual_label_col_index, input_text_col_index, document_files=None):
    """
    Predict on an entire dataset using the specified model.

    Args:
        report_name (str): Name for the prediction report.
        model_types (list): List of model types to use.
        model_id (str): The ID of the model to use for prediction.
        dataset_id (str): The ID of the dataset to predict on.
        actual_label_col_index (int): Index of the actual label column.
        input_text_col_index (int): The index of the input text column.
        document_files (list): List of paths to document files, if any.

    Returns:
        dict: Should return:
            - "predict_report_id": str - ID for the prediction report
            - "status": str - Status of the prediction job
            - "total_samples": int - Number of samples to process
    """
    print(f"Running predictions on dataset {dataset_id} with model {model_id}")
    
    # Should return prediction job information
    return {
        "predict_report_id": "report_123",
        "status": "processing",
        "total_samples": 1000
    }

def viewPredictions(self, predict_report_id, dataset_id, search_query, page_number):
    """
    View predictions for a given dataset and query.

    Args:
        predict_report_id (str): The ID of the prediction report.
        dataset_id (str): The ID of the dataset.
        search_query (str): The search query to filter predictions.
        page_number (int): The page number of the results.

    Returns:
        dict: Should return:
            - "predictions": List[dict] - Paginated prediction results
            - "total_results": int - Total number of matching results
            - "page_size": int - Number of results per page
            - "current_page": int - Current page number
            - "total_pages": int - Total number of pages
    """
    print(f"Viewing predictions for report {predict_report_id}, page {page_number}")
    
    # Should return paginated prediction results
    return {
        "predictions": [],
        "total_results": 0,
        "page_size": 50,
        "current_page": page_number,
        "total_pages": 0
    }

def _open_files(document_files):
    """Helper function to open document files"""
    if document_files is None:
        return {}, []
    
    # File opening logic would go here
    return {}, []

def _close_files(opened_files):
    """Helper function to close opened files"""
    for file in opened_files:
        if hasattr(file, 'close'):
            file.close()