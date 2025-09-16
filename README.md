# SCABG-LDT


### Key Features:
1. **SCABG Model**: A branch-based architecture that utilizes **Self-Calibrated Adaptive Band Gating** (SCABG) for multi-modal data.
2. **Cross-Branch Multi-Head Attention Fusion**: An enhancement to improve the integration of the multi-modal features.
3. **Dynamic Coarse/Fine Loss Weights**: Optimized by homoscedastic uncertainty.
4. **Focal Loss**: Optional focal loss to address class imbalance.
5. **Knowledge Distillation**: KD techniques for leveraging teacher models in both text (NLI) and image (CLIP-based).
6. **Robust Input Data Pipeline**: Support for different input formats like CSV, JSON, and JSONL for image and text data.

### Files Overview:

1. **train_upgraded.py**: 
    - Main script for training the multimodal sentiment model.
    - Supports custom dataset handling, training loops with various loss functions (including KD), and model saving.
    - Handles data balancing using oversampling and class weighting.

2. **multimodal_mh_upgraded.py**:
    - Defines the architecture of the multimodal model, including text and image encoding (using ViT or ResNet).
    - Implements SCABG with cross-branch attention and fusion techniques for better performance in multimodal tasks.
    - Includes support for focal loss, KD, and dynamic coarse/fine loss adjustments.

3. **enhanced_kd_te.py**:
    - Precomputes teacher logits for Knowledge Distillation.
    - Includes NLI-based teacher for emotion detection related to entities in the text and CLIP-based teacher for image-text matching.
    - Supports both NLI and CLIP inference for training with teacher models.

4. **dataset_v2.py**:
    - Defines the dataset loading pipeline for multimodal data (images and text).
    - Prepares the dataset by tokenizing the text and loading images while handling context and label information.
    - Supports flexible data input formats and normalization techniques for images.

### Installation:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/multimodal-sentiment-analysis.git
    cd multimodal-sentiment-analysis
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download necessary pretrained models:
    - **CLIP**: `openai/clip-vit-base-patch32`
    - **NLI model**: `roberta-large-mnli`

### Usage:

1. **Train the model**:
    - Prepare your dataset in CSV, JSON, or JSONL format with columns `image_path`, `text`, `entity`, `label` (and optionally `ctx_text`).
    - Run the training script:
      ```bash
      python train_upgraded.py \
        --train_csv /path/to/train_data.csv \
        --val_csv /path/to/val_data.csv \
        --img_root /path/to/images \
        --epochs 25 \
        --batch_size 32 \
        --lr 3e-5 \
        --save /path/to/save_model.pt
      ```

2. **Inference**:
    - Use the trained model to predict sentiment on new data:
      ```bash
      python inference.py --model /path/to/saved_model.pt --data /path/to/test_data.csv
      ```

### Configuration Options:

- `--epochs`: Number of training epochs (default: 25)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 3e-5)
- `--img_encoder`: Encoder for image data (default: `google/vit-base-patch16-224`)
- `--use_ctx`: Whether to use context information (default: False)
- `--lambda_kd`: Weight for knowledge distillation loss (default: 0.02)
- `--use_focal`: Whether to use focal loss (default: False)

### Example Usage for Knowledge Distillation (KD):

For knowledge distillation, the teacher models should be precomputed and provided during training:

```bash
python enhanced_kd_te.py \
  --in_path /path/to/data.csv \
  --out_path /path/to/output.json \
  --nli_model roberta-large-mnli \
  --clip_model openai/clip-vit-base-patch32 \
  --device cuda

