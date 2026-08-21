You are working on a college mini-project MVP for aerial image object detection using the DOTA dataset.

The goal is deliberately limited and practical. DO NOT over-engineer this into a production distributed ML platform.

## PROJECT GOAL

Build a complete, reproducible Python machine-learning pipeline that:

1. Uses the DOTA dataset.
2. Detects the standard 15 DOTA object categories.
3. Prepares/converts the dataset into the format required by the selected object-detection models.
4. Performs reasonable preprocessing.
5. Trains approximately 5 major object-detection model configurations.
6. Evaluates all candidate models using appropriate object-detection metrics.
7. Compares their results.
8. Automatically selects and saves the best-performing model.
9. Produces an inference-ready model artifact that the future Streamlit application can load.
10. Provides a simple inference script so we can test the final model on an image before integrating it into Streamlit.

This is a COLLEGE MVP. Prefer simplicity, reproducibility, and clarity over sophisticated infrastructure.

Do NOT introduce:

* Kubernetes
* Kafka
* RabbitMQ
* Microservices
* FastAPI
* distributed training
* cloud infrastructure
* complex MLOps platforms
* unnecessary databases
* model-serving infrastructure

Everything should work locally with Python.

---

# PART 1 — INSPECT THE DATASET FIRST

Before implementing the pipeline, inspect the existing DOTA dataset structure.

Do not assume the dataset layout.

Determine:

* image locations
* annotation locations
* annotation format
* image extensions
* class names
* number of images
* number of annotations
* image dimensions
* whether annotations are oriented bounding boxes/polygons
* whether the dataset is DOTA v1.0, v1.5, v2.0, or another DOTA-derived dataset

Create a small dataset inspection utility that reports these statistics.

If the actual dataset differs from the expected DOTA structure, adapt the preprocessing pipeline instead of corrupting or manually modifying the source dataset.

---

# PART 2 — STANDARD 15 DOTA CLASSES

Use these exact standard DOTA classes:

1. plane
2. ship
3. storage-tank
4. baseball-diamond
5. tennis-court
6. basketball-court
7. ground-track-field
8. harbor
9. bridge
10. large-vehicle
11. small-vehicle
12. helicopter
13. roundabout
14. soccer-ball-field
15. swimming-pool

Create one canonical class mapping in the project.

For example:

0 -> plane
1 -> ship
2 -> storage-tank
...
14 -> swimming-pool

Do not duplicate class mappings throughout the codebase.

---

# PART 3 — DUMMY KNOWLEDGE DOCUMENT

Before training the model, create a directory such as:

knowledge_base/

Inside it create one dummy base document containing information for ALL 15 classes.

Use JSON because this is an academic MVP and JSON will make the future RAG pipeline easy to demonstrate.

Example structure:

{
"plane": {
"title": "...",
"description": "...",
"information": "..."
},
"ship": {
"title": "...",
"description": "...",
"information": "..."
}
}

Continue for all 15 categories.

The information can be synthetic/dummy information because this document is only for demonstrating the RAG architecture.

Do not present the dummy information as authoritative real-world operational information.

Create approximately 2–4 useful paragraphs or structured fields for each category so that retrieval has enough text to work with.

The document should be clearly marked as:

"DEMO / SYNTHETIC KNOWLEDGE BASE"

Also create a README explaining that the document exists purely for the college demonstration.

---

# PART 4 — DATA SPLITTING

Create a configurable dataset split.

Default:

* train
* validation
* test

However, this project has explicit permission from the project guide to reuse images between train and test for the MVP demonstration if necessary.

Implement the split mechanism cleanly and document the limitation.

If the same images are reused, clearly report:

"WARNING: This evaluation is demonstrative and is not a leakage-free scientific benchmark."

Do not falsely present such metrics as generalization performance.

If the existing dataset already has official train/validation/test partitions, preserve them unless there is a strong reason not to.

---

# PART 5 — PREPROCESSING

Build a preprocessing/conversion pipeline suitable for the selected object-detection framework.

Important:

DOTA commonly uses oriented bounding boxes/polygon annotations.

Handle this correctly.

If the selected model requires a different annotation format, implement the necessary conversion.

Do not simply throw away the polygon information without documenting the consequence.

The preprocessing pipeline should:

* validate images
* validate annotations
* map classes
* convert annotations
* handle invalid/malformed annotations gracefully
* create train/validation/test directories
* generate a dataset configuration file
* produce useful preprocessing statistics

Include logging.

Do not modify the original raw dataset.

Use a generated/processed directory instead.

---

# PART 6 — MODEL CANDIDATES

Test approximately FIVE reasonable object-detection model configurations.

Do NOT test 20 models.

The purpose is to compare a small number of practical candidates.

Prefer modern YOLO-family object detectors or similarly practical detectors that are compatible with the dataset and local environment.

Possible candidates can include lightweight/small/medium variants such as:

* YOLO small
* YOLO medium
* YOLO large
* another YOLO-family configuration
* one additional major detector if it is practical in the environment

Before choosing the exact five, inspect the available hardware and installed libraries.

The models must be realistically trainable on the available hardware.

If GPU memory is limited, automatically use smaller models/image sizes/batch sizes rather than making the project unusable.

The goal is not to achieve state-of-the-art DOTA performance.

The goal is to identify a reasonable model for this MVP.

---

# PART 7 — TRAINING CONFIGURATION

Create centralized configuration.

For example:

configs/

with YAML or Python configuration containing:

* dataset path
* image size
* epochs
* batch size
* learning rate
* device
* workers
* random seed
* output directory
* model candidates

Do not hard-code these values across scripts.

Allow the user to run:

python train.py

or an equivalent simple command.

Also provide a way to train only one model for debugging.

Example concept:

python train.py --model candidate_1

---

# PART 8 — EVALUATION

Evaluate every trained candidate.

At minimum capture:

* precision
* recall
* mAP@0.5
* mAP@0.5:0.95
* inference time if practical
* model size
* training duration

Also generate per-class performance where supported.

The 15 categories should be visible individually.

Create a comparison table such as:

Model | Precision | Recall | mAP50 | mAP50-95 | Size | Inference Time

Save the results as CSV/JSON.

Also generate simple visualizations if practical.

---

# PART 9 — BEST MODEL SELECTION

Create automatic model selection logic.

The primary selection metric should be:

mAP@0.5:0.95

Use other metrics as secondary considerations.

Do not simply choose the model with the highest training accuracy.

The selected model should be copied/saved into a stable location such as:

artifacts/
best_model/
model.pt
metadata.json
class_names.json

metadata.json should contain:

* model name
* training configuration
* dataset information
* evaluation metrics
* date/time
* class mapping
* image size
* framework version if easily available

This artifact will later be consumed by the Streamlit application.

---

# PART 10 — INFERENCE SCRIPT

Create:

inference.py

It should allow:

python inference.py --image path/to/image.jpg

and produce:

* detected classes
* confidence scores
* bounding boxes
* annotated image

If the detector supports oriented bounding boxes, preserve them where practical.

Also save the annotated result.

Example output:

Detected Objects:

* plane: 0.94
* small-vehicle: 0.87
* small-vehicle: 0.82

Do not assume there is only one object in an image.

The future Streamlit application should be able to use the same inference function.

Therefore structure inference as a reusable Python function, not only a command-line script.

Example conceptual interface:

predict(image) -> detection_results

---

# PART 11 — PROJECT STRUCTURE

Create a clean but simple structure similar to:

project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── knowledge_base/
│   └── base_knowledge.json
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── data/
│   │   ├── inspect.py
│   │   ├── preprocess.py
│   │   └── split.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── select_best.py
│   │
│   ├── inference/
│   │   └── predictor.py
│   │
│   └── utils/
│
├── artifacts/
│   └── best_model/
│
├── reports/
│   ├── model_comparison.csv
│   └── evaluation/
│
├── scripts/
│
├── requirements.txt
└── README.md

You may adjust the structure if necessary, but keep it simple.

---

# PART 12 — REPRODUCIBILITY

Set random seeds where applicable.

Create:

requirements.txt

with the required dependencies.

Document:

* Python version
* GPU requirements
* CPU fallback
* installation instructions
* dataset placement
* preprocessing command
* training command
* evaluation command
* inference command

A faculty member should be able to understand the pipeline by reading the README.

---

# PART 13 — ERROR HANDLING

Handle common problems gracefully:

* missing dataset
* incorrect dataset path
* malformed annotations
* missing model weights
* unsupported image
* insufficient GPU memory
* missing Python dependency

Give clear error messages.

Do not silently ignore errors.

---

# PART 14 — FINAL ACCEPTANCE CRITERIA

The phase is COMPLETE only when:

1. The DOTA dataset has been inspected.
2. The 15 classes are correctly mapped.
3. The synthetic base JSON document exists for all 15 classes.
4. The preprocessing pipeline works.
5. At least five practical model candidates can be trained/evaluated, subject to hardware constraints.
6. Evaluation metrics are collected.
7. Models are compared.
8. The best model is automatically selected.
9. The best model is saved as an inference-ready artifact.
10. `predictor.py` can load the saved model.
11. An image can be passed through the predictor.
12. Detected classes and confidence scores are returned.
13. An annotated image can be generated.
14. The project can be reproduced using the README.

IMPORTANT:

Do not start building the Streamlit RAG application in this phase.

The final output of this phase is:

DOTA dataset
→ preprocessing
→ model training
→ evaluation
→ best model
→ reusable predictor
→ synthetic 15-class knowledge document

Stop after this phase is fully working.
