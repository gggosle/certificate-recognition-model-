# Model Testing Guide

This guide explains how to test the fine-tuned LayoutLMv3 model for certificate information extraction.

## Prerequisites

1. Python 3.7+
2. Required Python packages (install with `pip install -r requirements.txt`)
3. Fine-tuned model in the `my_custom_model` directory
4. Test dataset in the following structure:
   ```
   certificate-dataset-v1/
   └── test/
       ├── images/      # Test images (.jpg, .png)
       └── labels/      # Corresponding LabelMe JSON annotations
   ```

## Running the Tests

1. **Run the test script**:
   ```bash
   python test_model.py
   ```

2. **Expected Output**:
   - The script will process each test image and generate:
     - Visualizations with bounding boxes and labels in `./test_results/visualizations/`
     - A JSON file with all predictions in `./test_results/predictions.json`

## Understanding the Results

### Visualizations
- Each test image will be saved with predicted bounding boxes and labels overlaid
- Box colors indicate different entity types (NAME, COURSE_NAME, ISSUER, etc.)
- The text above each box shows the predicted label and the recognized text

### Predictions JSON

The `predictions.json` file contains detailed prediction results in the following format:

```json
[
  {
    "image_path": "path/to/image.jpg",
    "entities": [
      {
        "label": "NAME",
        "text": "John Doe",
        "boxes": [[x1, y1, x2, y2], ...]
      },
      ...
    ]
  },
  ...
]
```

## Interpreting the Results

1. **Entity Extraction**: The model should correctly identify and classify text regions into:
   - `NAME`: Name of the certificate recipient
   - `COURSE_NAME`: Name of the course or program
   - `ISSUER`: Organization issuing the certificate
   - `OTHER`: Other relevant information

2. **Common Issues**:
   - Misclassified entities: Check if the text is visually similar to other entities
   - Missed entities: May occur with unusual fonts or layouts
   - Overlapping boxes: Could indicate OCR or model confidence issues

## Next Steps

1. Review the visualizations to identify common error patterns
2. For poor performance, consider:
   - Adding more training examples for problematic cases
   - Adjusting the model's confidence threshold
   - Post-processing the results to handle common error patterns

## Troubleshooting

- If you encounter memory issues, reduce the batch size in the test script
- Ensure all test images have corresponding JSON annotations in the labels directory
- Check that the label names in your test data match those used during training
