import asyncio
import os
from kyc_pipeline import DocumentPipeline

DOC_MAPPING = {
    '1': ('AADHAAR', 'Aadhaar Card'),
    '2': ('PAN', 'PAN Card'),
    '3': ('VOTER_ID', 'Voter ID Card'),
    '4': ('DRIVING_LICENSE', 'Driving License'),
    '5': ('PASSPORT', 'Passport')
}

def print_results(result):
    pipeline = DocumentPipeline()
    print("\n" + "="*40)
    print(" RAW OCR EXTRACTED TEXT ")
    print("="*40)
    
    clean_text = "\n".join([line for line in result.get("extracted_text", "").split("\n") if line.strip()])
    print(clean_text if clean_text else "[No Text Read]")
    print("="*40)
    
    print("\n--- Final KYC Status ---")
    print(f"Status          : {result.get('status', 'FAILED')}")
    print(f"Detected Type   : {pipeline.get_printable_name(result.get('actual_type'))}")
    
    if result.get("status") == "SUCCESS":
        print(f"Extracted ID    : {result.get('extracted_id')}")
        print(f"Text Masked     : {result.get('masked_id')}")
        print(f"Image Masked    : Saved to -> {result.get('masked_image_file')}")
    else:
        print(f"Failure Reason  : {result.get('message')}")
    print("-" * 24 + "\n")

def run_interactive_pipeline():
    pipeline = DocumentPipeline()
    
    while True:
        print("\n--- Document Classifier, Verifier & Masker ---")
        for key, (code, name) in DOC_MAPPING.items():
            print(f"{key}. {name}")
        print("0. Exit")
        
        selection = input("Select intended document type (0-5): ").strip()
        if selection == '0':
            print("Exiting pipeline. Goodbye!")
            break
        if selection not in DOC_MAPPING:
            print("Invalid option. Please choose a valid target number.")
            continue
            
        intended_code, intended_name = DOC_MAPPING[selection]
        file_path = input(f"Enter file path for {intended_name} (Supports Images & PDFs): ").strip()
        
        if not os.path.isfile(file_path):
            print(f"Error: File '{file_path}' not found. Please recheck your targeted paths.")
            continue
            
        print(f"\nProcessing {file_path} (Evaluating document data layers)...")
        try:
            result = asyncio.run(
                pipeline.process_and_verify(file_path, intended_code)
            )
            print_results(result)
        except Exception as e:
            print(f"Execution Error Encountered: {e}")
            
if __name__ == "__main__":
    run_interactive_pipeline()