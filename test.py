import os
import torch
from PIL import Image
from transformers import LayoutLMv3Processor
from transformers import LayoutLMv3ForTokenClassification
from ocr_init import get_ocr_engine
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import CertificateDataset

# Configuration
MODEL_PATH = "./my_custom_model"
OUTPUT_IMAGE = "test_output.png"
LABEL_LIST = ["O", "NAME", "COURSE_NAME", "ISSUER", "OTHER"]

def load_model_and_processor():
    """Load the fine-tuned model and processor."""
    print("Loading model and processor...")
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(LABEL_LIST)
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor

def extract_ocr_lines(ocr_result, min_score=0.6):
    texts = ocr_result["rec_texts"]
    scores = ocr_result["rec_scores"]
    polys  = ocr_result["rec_polys"]

    items = []
    for text, score, poly in zip(texts, scores, polys):
        if not text.strip():
            continue
        if score < min_score:
            continue

        # Convert polygon → bbox (x0, y0, x1, y1)
        xs = poly[:, 0]
        ys = poly[:, 1]
        bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

        items.append({
            "text": text,
            "score": float(score),
            "bbox": bbox,
            "poly": poly.tolist(),
        })

    return items

def split_line_bbox_weighted(bbox, words):
        x1, y1, x2, y2 = bbox
        total_chars = sum(len(w) for w in words)
        cur_x = x1

        boxes = []
        for w in words:
            w_width = (len(w) / total_chars) * (x2 - x1)
            wx1 = int(cur_x)
            wx2 = int(cur_x + w_width)
            boxes.append([wx1, y1, wx2, y2])
            cur_x = wx2

        return boxes

def process_single_image(image_path, model, processor):
    """Process a single image through the model."""
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ocr_engine = get_ocr_engine()
    
    # Get OCR results
    ocr_result = ocr_engine.predict(image_path)
    words = []
    word_boxes = []

    for res in ocr_result:
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        polys = res["rec_polys"]

        for text, score, poly in zip(texts, scores, polys):
            if not text.strip() or score < 0.6:  # Skip empty text and low confidence detections
                continue

            # Convert polygon to bbox [x0, y0, x1, y1]
            xs = poly[:, 0]
            ys = poly[:, 1]
            bbox = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max())
            ]

            # Normalize boxes to 0-1000 scale
            norm_bbox = [
                int(1000 * (bbox[0] / width)),
                int(1000 * (bbox[1] / height)),
                int(1000 * (bbox[2] / width)),
                int(1000 * (bbox[3] / height)),
            ]

            

            # Split text into words and duplicate the bbox for each word
            line_words = text.split()

            words.extend(line_words)
            word_boxes.extend(split_line_bbox_weighted(norm_bbox, line_words))

    # Print debug info
    print(f"Words: {words}")
    print(f"Number of words: {len(words)}")
    print(f"Number of boxes: {len(word_boxes)}")

    if not words:
        raise ValueError("No text detected in the image or all text was filtered out")
    
    # Prepare model inputs
    encoding = processor(
        image,
        text=words,
        boxes=word_boxes,  # Now one box per word
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for k, v in encoding.items():
        encoding[k] = v.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Process outputs
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    return image, words, word_boxes, predictions


def visualize_results(image, words, boxes, predictions, output_path):
    """Visualize and save the prediction results."""
    fig, ax = plt.subplots(1, figsize=(12, 15))
    ax.imshow(image)
    width, height = image.size
    
    for word, box, pred_id in zip(words, boxes, predictions):
        if pred_id == 0:  # Skip 'O' class
            continue
        denorm_bbox = [
            int((box[0] / 1000) * width),
            int((box[1] / 1000) * height),
            int((box[2] / 1000) * width),
            int((box[3] / 1000) * height),
            ]    
        label = LABEL_LIST[pred_id]
        x1, y1, x2, y2 = denorm_bbox
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        plt.text(
            x1, y1 - 2,
            f"{label}: {word}",
            color='red', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    # Initialize
    torch.cuda.empty_cache()
    model, processor = load_model_and_processor()
    
    # Get input image path
    image_path = input("Enter the path to the image file: ").strip('"')
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    # Process image
    print("Processing image...")
    image, words, boxes, predictions = process_single_image(image_path, model, processor)
    print(predictions)
    # Save results
    output_path = os.path.join(os.path.dirname(image_path), OUTPUT_IMAGE)
    visualize_results(image, words, boxes, predictions, output_path)
    
    print(f"\nProcessing complete! Results saved to: {output_path}")

if __name__ == "__main__":
    main()