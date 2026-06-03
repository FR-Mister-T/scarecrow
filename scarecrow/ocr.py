import cv2

_PLATE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"


def load_reader():
    """Construct a RapidOCR reader. Requires the [ocr] extra."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def crop_for_ocr(img, bbox, pad=0.15):
    """Crop the plate bbox with a margin around it."""
    x, y, w, h = bbox
    h_img, w_img = img.shape[:2]
    cx1 = max(0, x - int(w * pad))
    cy1 = max(0, y - int(h * pad))
    cx2 = min(w_img, x + w + int(w * pad))
    cy2 = min(h_img, y + h + int(h * pad))
    return img[cy1:cy2, cx1:cx2]


def read_plate(reader, crop):
    """Read plate text from a crop. Selects the largest text region."""
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    result, _ = reader(bgr)
    if not result:
        return ""
    # item: (bbox, text, conf)
    # bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    best = max(result, key=lambda r: (r[0][2][1] - r[0][0][1]) * (r[0][1][0] - r[0][0][0]))
    text = best[1]
    return "".join(c for c in text.strip().upper() if c in _PLATE_CHARS)
