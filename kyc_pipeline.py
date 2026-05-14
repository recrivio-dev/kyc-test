import os
import re
import cv2
import numpy as np
import pypdfium2 as pdfium
import pytesseract
from pytesseract import Output

class DocumentPipeline:
    def __init__(self):
        self.patterns = {
            "PAN": r'[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            "AADHAAR": r'\b\d{4}[\s\-\.]?\d{4}[\s\-\.]?\d{4}\b',
            "VOTER_ID": r'[A-Z]{3}[\s\-\.\:]*[0-9]{7}\b',
            "PASSPORT": r'[A-Z]{1,2}[\s\-\.\:]*[0-9]{6,7}\b',
            "DRIVING_LICENSE": r'[A-Z]{2}[A-Z0-9]{2}[\s\-\.\:]*[0-9]{7,11}\b'
        }
        
        self.keywords = {
            "PAN": ["INCOME TAX", "PERMANENT ACCOUNT", "PAN CARD"],
            "AADHAAR": ["UIDAI", "MERA AADHAAR", "VID :"],
            "VOTER_ID": ["ELECTION COMMISSION", "ELECTOR", "EPIC", "FACSIMILE"],
            "PASSPORT": ["PASSPORT", "REPUBLIC OF INDIA", "P<IND"],
            "DRIVING_LICENSE": ["DRIVING LICENCE", "DRIVING LICENSE", "MOTOR DRIVING", "AUTHORISATION", "DL NO"]
        }

    def convert_pdf_to_image(self, pdf_path):
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[0]
        pil_image = page.render(scale=4).to_pil()
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return open_cv_image

    def load_document_image(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot locate document at: {file_path}")
            
        ext = file_path.lower().split('.')[-1]
        if ext == 'pdf':
            return self.convert_pdf_to_image(file_path)
        else:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Failed to read image data from: {file_path}. File may be corrupted.")
            return img

    def _preprocess_image(self, image):
        scaled = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return scaled, thresh

    def extract_and_orient(self, file_path):
        original_img = self.load_document_image(file_path)
        current_img = original_img.copy()
        
        for angle in [0, 90, 180, 270]:
            if angle > 0:
                current_img = cv2.rotate(current_img, cv2.ROTATE_90_CLOCKWISE)
                
            scaled_img, thresh_img = self._preprocess_image(current_img)
            text = pytesseract.image_to_string(thresh_img, config='--psm 3').upper()
            
            if self.classify_document(text) != "UNKNOWN":
                return text, scaled_img, thresh_img

        scaled_img, thresh_img = self._preprocess_image(original_img)
        return pytesseract.image_to_string(thresh_img, config='--psm 3').upper(), scaled_img, thresh_img

    def classify_document(self, text):
        scores = {doc_type: 0 for doc_type in self.keywords}
        for doc_type, kw_list in self.keywords.items():
            for kw in kw_list:
                if kw in text:
                    scores[doc_type] += 3
                    
        for doc_type, pattern in self.patterns.items():
            if re.search(pattern, text):
                scores[doc_type] += 5

        max_doc = max(scores, key=scores.get)
        return max_doc if scores[max_doc] > 0 else "UNKNOWN"

    def get_printable_name(self, doc_type_code):
        names = {
            "PAN": "PAN Card",
            "AADHAAR": "Aadhaar Card",
            "VOTER_ID": "Voter ID Card",
            "PASSPORT": "Passport",
            "DRIVING_LICENSE": "Driving License"
        }
        return names.get(doc_type_code, "Unknown Document")

    def verify_and_extract(self, text, doc_type):
        if doc_type not in self.patterns:
            return None
        matches = re.findall(self.patterns[doc_type], text)
        return matches[0] if matches else None

    def mask_id(self, id_string, doc_type):
        if not id_string:
            return None
        clean_id = re.sub(r'[\s\-\.\:]', '', id_string)
        
        if doc_type == "AADHAAR":
            return f"[Aadhaar Redacted]-{clean_id[-4:]}"
        elif len(clean_id) > 4:
            return "X" * (len(clean_id) - 4) + clean_id[-4:]
        return "[Redacted]"

    def create_masked_image(self, thresh_img, color_img, id_string, output_path):
        if not id_string:
            return None

        tokens = [t for t in re.split(r'[\s\-\.\:]', id_string) if len(t) >= 2]
        clean_full_id = re.sub(r'[\s\-\.\:]', '', id_string)
        
        ocr_data = pytesseract.image_to_data(thresh_img, output_type=Output.DICT, config='--psm 3')
        n_boxes = len(ocr_data['text'])
        
        masked_img = color_img.copy()
        boxes_drawn = 0

        for i in range(n_boxes):
            word = ocr_data['text'][i].upper()
            clean_word = re.sub(r'[\s\-\.\:]', '', word)
            
            if len(clean_word) >= 3 and (clean_word in clean_full_id or any(t in clean_word for t in tokens)):
                # Avoid redacting the trailing 4 characters on physical Aadhaar images
                if clean_full_id[-4:] in clean_word and len(clean_word) <= 5:
                    continue
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                cv2.rectangle(masked_img, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 0, 0), -1)
                boxes_drawn += 1

        # Save output consistently as JPEG
        base_name = output_path.rsplit('.', 1)[0]
        final_output_path = f"{base_name}.jpg" if output_path.lower().endswith('.pdf') else output_path
        
        cv2.imwrite(final_output_path, masked_img)
        return final_output_path if boxes_drawn > 0 else None

    def process_and_verify(self, file_path, intended_doc_type):
        text, oriented_img, thresh_img = self.extract_and_orient(file_path)
        actual_doc_type = self.classify_document(text)
        
        result = {
            "extracted_text": text.strip(),
            "actual_type": actual_doc_type
        }

        if actual_doc_type == intended_doc_type:
            id_number = self.verify_and_extract(text, actual_doc_type)
            if id_number:
                filename = os.path.basename(file_path)
                # Output directly to sample-docs for clean state mapping
                masked_image_path = f"sample-docs/masked_{filename}"
                saved_path = self.create_masked_image(thresh_img, oriented_img, id_number, masked_image_path)
                
                result.update({
                    "status": "SUCCESS",
                    "message": "Valid ID pattern found and masked.",
                    "extracted_id": id_number,
                    "masked_id": self.mask_id(id_number, actual_doc_type),
                    "masked_image_file": saved_path or "Bounding box mapping missed; plain text masked successfully."
                })
            else:
                result.update({
                    "status": "FAILED",
                    "message": f"Verified document type as {self.get_printable_name(actual_doc_type)}, but OCR failed to extract the exact ID pattern."
                })
        else:
            if actual_doc_type == "UNKNOWN":
                result.update({
                    "status": "FAILED",
                    "message": "Could not recognize document type. Image quality might be too low."
                })
            else:
                intended_readable = self.get_printable_name(intended_doc_type)
                actual_readable = self.get_printable_name(actual_doc_type)
                result.update({
                    "status": "FAILED",
                    "message": f"Type Mismatch: Expected {intended_readable}, detected {actual_readable}."
                })
        return result