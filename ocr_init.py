from paddleocr import PaddleOCR
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            lang="uk", # Specify French recognition model with the lang parameter
            use_doc_orientation_classify=False, # Disable document orientation classification model
            use_doc_unwarping=False, # Disable text image unwarping model
            use_textline_orientation=False, # Disable text line orientation classification model
        )
    return _ocr_engine