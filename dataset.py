import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor
from ocr_init import get_ocr_engine
import os

import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor
import os

class CertificateDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor, label_list):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.label_list = label_list
        self.ocr_engine = get_ocr_engine()
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.label2id = {label: i for i, label in enumerate(label_list)}

    def __len__(self):
        return len(self.image_files)


    def _load_labelme_boxes(self, json_path, width, height):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error loading JSON: {json_path}")
            return [] # Handle missing JSONs safely

        ground_truth = []
        for shape in data.get('shapes', []): # Safety .get
            label = shape['label']
            if label in ["other", "Text"]:
                continue

            points = shape['points']
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]

            x1, y1 = min(x_vals), min(y_vals)
            x2, y2 = max(x_vals), max(y_vals)

            ground_truth.append({
                "label": label.upper(),
                "box": [x1, y1, x2, y2]
            })
        return ground_truth

    def box_overlap_ratio(self, boxA, boxB):
        # 1. Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # 2. Compute the area of intersection
        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        # 3. Compute the area of both boxes
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # 4. Find the smaller area to use as the divisor
        smaller_area = min(areaA, areaB)

        # Handle potential division by zero if a box has no area
        if smaller_area <= 0:
            return 0.0

        # 5. Return ratio of intersection over the smaller box
        return inter_area / smaller_area

    def split_line_bbox_weighted(self, bbox, words):
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

    def __getitem__(self, idx):
        try:
            return self._process_item(idx)
        except Exception as e:
            print(f"Error processing {self.image_files[idx]}: {e}")

    def _process_item(self, idx):
        file_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        json_path = os.path.join(self.label_dir, base_name + ".json")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        ocr_result = self.ocr_engine.predict(image_path)
        words = []
        word_boxes = []
        word_pixel_boxes = []

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
                # Append the same bbox for each word in the line
                word_boxes.extend([norm_bbox] * len(line_words))
                word_pixel_boxes.extend([bbox] * len(line_words))

        # Print debug info
        print(f"Words: {words}")
        print(f"Number of words: {len(words)}")
        print(f"Number of boxes: {len(word_boxes)}")

        if not words:
            raise ValueError("No text detected in the image or all text was filtered out")

        # 2. Load Ground Truth
        ground_truth = self._load_labelme_boxes(json_path, width, height)

        # 3. Match
        ner_tags = []
        for word_box in word_pixel_boxes:
            label_found = "O"
            for gt in ground_truth:
                if self.box_overlap_ratio(word_box, gt['box']) > 0.6:
                    label_found = gt['label']
                    break
            ner_tags.append(self.label2id.get(label_found, 0))

        # Handle Empty OCR result (e.g. blank page) to prevent crash
        if not words:
            words = ["Empty"]
            word_boxes = [[0,0,1,1]]
            ner_tags = [0]

        #debug_draw(image, word_pixel_boxes, ground_truth)
        print(f"NER Tags: {ner_tags}")


        encoding = self.processor(
            image,
            words,
            boxes=word_boxes,
            word_labels=ner_tags,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {k: v.squeeze() for k, v in encoding.items()}

from PIL import ImageDraw

def debug_draw(image, ocr_boxes, gt_boxes):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # OCR boxes → RED
    for b in ocr_boxes:
        draw.rectangle(b, outline="red", width=2)

    # GT boxes → GREEN
    for gt in gt_boxes:
        draw.rectangle(gt["box"], outline="green", width=3)
    img.show()

def main():
    IMAGE_DIR = "./certificate-dataset-v1/data/images"
    LABEL_DIR = "./certificate-dataset-v1/data/labels"
    processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
    )

    dataset = CertificateDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        processor=processor,
        label_list=["O", "COURSE_NAME", "ISSUER", "SIGNATURE"]
    )

    # Force-debug ONE sample
    idx = 3
    file_name = dataset.image_files[idx]
    image_path = os.path.join(IMAGE_DIR, file_name)
    json_path = os.path.join(LABEL_DIR, file_name.replace(".jpg", ".json").replace(".png", ".json").replace(".jpeg", ".json"))

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    gt = dataset._load_labelme_boxes(json_path, width, height)

    ocr_result = dataset.ocr_engine.predict(image_path)

    ocr_pixel_boxes = []
    for res in ocr_result:
        for poly in res["rec_polys"]:
            xs = poly[:, 0]
            ys = poly[:, 1]
            ocr_pixel_boxes.append([
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max())
            ])

    print("DEBUG:")
    print("Image size:", width, height)
    print("OCR boxes (pixel):", ocr_pixel_boxes[:5])
    print("GT boxes (pixel):", [g["box"] for g in gt])
    dataset._process_item(idx)

    

if __name__ == "__main__":
    main()


