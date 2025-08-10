
from core import Anote, NLPTask, ModelType, EvaluationMetric
from time import sleep
import json

api_key = 'INSERT_API_KEY_HERE'

# Initialize the Anote class
anote = Anote(api_key)

# Dataset preparation - you would need to prepare your data in JSON format
# Example structure for the JSON files:
"""
{
  "images": [
    {
      "id": 0,
      "file_path": "path/to/image1.jpg",
      "annotations": [
        {
          "bbox": [x1, y1, x2, y2],
          "label": "A5"
        }
      ]
    }
  ]
}
"""

# Upload datasets
print("Uploading datasets...")

# Upload training dataset
train_upload_response = anote.upload(
    dataset_name="aircraft_train",
    data_path="./data/train_annotations.json",
    split="train"
)
print("Train upload response:", train_upload_response)

# Upload validation dataset
val_upload_response = anote.upload(
    dataset_name="aircraft_validation", 
    data_path="./data/val_annotations.json",
    split="validation"
)
print("Validation upload response:", val_upload_response)

# Upload test dataset
test_upload_response = anote.upload(
    dataset_name="aircraft_test",
    data_path="./data/test_annotations.json", 
    split="test"
)
print("Test upload response:", test_upload_response)

# Train models with different architectures
print("\nTraining models...")

# Train Faster R-CNN model
faster_rcnn_response = anote.train(
    task_type=NLPTask.OBJECT_DETECTION,  # Using 5 for object detection
    model_type="faster_rcnn",
    train_dataset="./data/train_annotations.json",
    validation_dataset="./data/val_annotations.json"
)
faster_rcnn_model_id = faster_rcnn_response.get("model_id")
print(f"Faster R-CNN model ID: {faster_rcnn_model_id}")

# Train YOLOv8 model
yolo_response = anote.train(
    task_type=NLPTask.OBJECT_DETECTION,
    model_type="yolov8", 
    train_dataset="./data/train_annotations.json",
    validation_dataset="./data/val_annotations.json"
)
yolo_model_id = yolo_response.get("model_id")
print(f"YOLOv8 model ID: {yolo_model_id}")

# Train Grounding DINO model
dino_response = anote.train(
    task_type=NLPTask.OBJECT_DETECTION,
    model_type="grounding_dino",
    train_dataset="./data/train_annotations.json", 
    validation_dataset="./data/val_annotations.json"
)
dino_model_id = dino_response.get("model_id")
print(f"Grounding DINO model ID: {dino_model_id}")

# Wait for training to complete
print("\nWaiting for training to complete...")
models_to_check = [
    ("Faster R-CNN", faster_rcnn_model_id),
    ("YOLOv8", yolo_model_id), 
    ("Grounding DINO", dino_model_id)
]

for model_name, model_id in models_to_check:
    if model_id:
        while True:
            train_status_response = anote.checkStatus(model_id=model_id)
            if train_status_response.get("isComplete") == True:
                print(f"{model_name} training complete...")
                break
            else:
                print(f"Waiting for {model_name} training...")
                sleep(10)

# Define class labels for your dataset
class_labels = ["A5", "A9", "A10", "background"]  # Update with your actual classes

# Make predictions with different models
print("\nGenerating predictions...")

# Faster R-CNN predictions
faster_rcnn_predictions = anote.predict(
    model_type="faster_rcnn",
    test_data="./data/test_images/",  # Path to test images directory
    labels=class_labels,
    model_id=faster_rcnn_model_id,
    confidence_threshold=0.5
)

# YOLOv8 predictions  
yolo_predictions = anote.predict(
    model_type="yolov8",
    test_data="./data/test_images/",
    labels=class_labels,
    model_id=yolo_model_id,
    confidence_threshold=0.5
)

# Grounding DINO predictions
dino_predictions = anote.predict(
    model_type="grounding_dino", 
    test_data="./data/test_images/",
    labels=class_labels,
    model_id=dino_model_id,
    confidence_threshold=0.5
)

print("Faster R-CNN Predictions:", len(faster_rcnn_predictions))
print("YOLOv8 Predictions:", len(yolo_predictions))
print("Grounding DINO Predictions:", len(dino_predictions))

# Save predictions to files for evaluation
with open("faster_rcnn_predictions.json", "w") as f:
    json.dump(faster_rcnn_predictions, f, indent=2)

with open("yolo_predictions.json", "w") as f:
    json.dump(yolo_predictions, f, indent=2)
    
with open("dino_predictions.json", "w") as f:
    json.dump(dino_predictions, f, indent=2)

# Evaluate models
print("\nEvaluating models...")

# Evaluate Faster R-CNN
faster_rcnn_eval_results = anote.evaluate(
    ground_truths="./data/test_annotations.json",
    predictions="faster_rcnn_predictions.json"
)
print("Faster R-CNN Evaluation Results:", faster_rcnn_eval_results)

# Evaluate YOLOv8
yolo_eval_results = anote.evaluate(
    ground_truths="./data/test_annotations.json", 
    predictions="yolo_predictions.json"
)
print("YOLOv8 Evaluation Results:", yolo_eval_results)

# Evaluate Grounding DINO
dino_eval_results = anote.evaluate(
    ground_truths="./data/test_annotations.json",
    predictions="dino_predictions.json" 
)
print("Grounding DINO Evaluation Results:", dino_eval_results)

# Compare results
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(f"Faster R-CNN - Predictions: {len(faster_rcnn_predictions)}")
print(f"YOLOv8 - Predictions: {len(yolo_predictions)}")
print(f"Grounding DINO - Predictions: {len(dino_predictions)}")
print("="*50)

# The evaluation will generate:
# - confusion_matrix.png: Visual confusion matrix
# - metrics.csv: Detailed metrics (Precision, Recall, Accuracy, F1-score, mIoU, mAP)
print("\nEvaluation complete! Check the generated files:")
print("- confusion_matrix.png")
print("- metrics.csv")

