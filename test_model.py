import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from dataset import CertificateDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# Configuration
TEST_IMAGE_DIR = "./certificate-dataset-v1/test/images"
TEST_LABEL_DIR = "./certificate-dataset-v1/test/labels"
MODEL_PATH = "./my_custom_model"
OUTPUT_DIR = "./test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define labels (must match training)
label_list = ["O", "NAME", "COURSE_NAME", "ISSUER", "OTHER"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

def load_model_and_processor():
    """Load the fine-tuned model and processor."""
    print("Loading model and processor...")
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_PATH,
        id2label=id2label,
        label2id=label2id
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor

def run_inference(model, processor, test_dataset):
    """Run inference on test dataset and collect predictions."""
    model.eval()
    predictions = []
    
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference"):
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            bbox = batch["bbox"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            pixel_values = batch["pixel_values"].to(model.device)
            
            # Get predictions
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            # Get predicted labels
            predicted_labels = torch.argmax(outputs.logits, dim=2)
            
            # Store results
            for i in range(len(batch["image"])):
                preds = predicted_labels[i].cpu().numpy()
                words = batch["tokens"][i]
                boxes = batch["bbox"][i].cpu().numpy()
                
                # Filter out padding tokens and special tokens
                valid_indices = (input_ids[i] != processor.tokenizer.pad_token_id) & \
                              (input_ids[i] != processor.tokenizer.cls_token_id) & \
                              (input_ids[i] != processor.tokenizer.sep_token_id)
                
                valid_preds = preds[valid_indices]
                valid_words = [words[j] for j in range(len(words)) if valid_indices[j]]
                valid_boxes = [boxes[j] for j in range(len(boxes)) if valid_indices[j]]
                
                predictions.append({
                    "image_path": batch["image_path"][i],
                    "words": valid_words,
                    "boxes": valid_boxes,
                    "predictions": valid_preds.tolist()
                })
    
    return predictions

def visualize_results(predictions, output_dir):
    """Visualize and save prediction results."""
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    for pred in predictions:
        image_path = pred["image_path"]
        image = Image.open(image_path).convert("RGB")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 15))
        ax.imshow(image)
        
        # Draw bounding boxes and labels
        for word, box, label_id in zip(pred["words"], pred["boxes"], pred["predictions"]):
            if label_id == 0:  # Skip 'O' class
                continue
                
            label = id2label.get(label_id, "UNK")
            x1, y1, x2, y2 = box
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label text
            plt.text(
                x1, y1 - 2,
                f"{label}: {word}",
                color='red', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        
        # Save visualization
        output_path = os.path.join(
            output_dir,
            "visualizations",
            os.path.basename(image_path)
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

def save_predictions(predictions, output_dir):
    """Save predictions to a JSON file."""
    output = []
    
    for pred in predictions:
        result = {
            "image_path": pred["image_path"],
            "entities": []
        }
        
        current_entity = None
        current_text = []
        current_boxes = []
        
        for word, box, label_id in zip(pred["words"], pred["boxes"], pred["predictions"]):
            if label_id == 0:  # 'O' class
                if current_entity is not None:
                    result["entities"].append({
                        "label": current_entity,
                        "text": " ".join(current_text),
                        "boxes": current_boxes
                    })
                    current_entity = None
                    current_text = []
                    current_boxes = []
                continue
                
            label = id2label.get(label_id, "UNK")
            if label != current_entity:
                if current_entity is not None:
                    result["entities"].append({
                        "label": current_entity,
                        "text": " ".join(current_text),
                        "boxes": current_boxes
                    })
                current_entity = label
                current_text = [word]
                current_boxes = [box]
            else:
                current_text.append(word)
                current_boxes.append(box)
        
        # Add the last entity if exists
        if current_entity is not None:
            result["entities"].append({
                "label": current_entity,
                "text": " ".join(current_text),
                "boxes": current_boxes
            })
        
        output.append(result)
    
    # Save to JSON
    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(output, f, indent=2)

def main():
    # Load model and processor
    model, processor = load_model_and_processor()
    
    # Create test dataset
    print("Preparing test dataset...")
    test_dataset = CertificateDataset(
        image_dir=TEST_IMAGE_DIR,
        label_dir=TEST_LABEL_DIR,
        processor=processor,
        label_list=label_list
    )
    
    # Run inference
    print("Running inference on test set...")
    predictions = run_inference(model, processor, test_dataset)
    
    # Save results
    print("Saving results...")
    save_predictions(predictions, OUTPUT_DIR)
    visualize_results(predictions, OUTPUT_DIR)
    
    print(f"\nTest completed! Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
