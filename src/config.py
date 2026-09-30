import os

# 1. Proje Kök Dizini ve Yol Hesaplamaları 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 2. Model Önbellek Kilitleri 
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
os.environ["TORCH_HOME"] = os.path.join(BASE_DIR, "torch_cache")

# 3. Model İsimleri (Hugging Face ve OpenAI Modelleri)
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CLIP_MODEL_NAME = "ViT-B/32"

# 4. Algoritma Parametreleri
HALUCINATION_THRESHOLD = 0.32  # Bu skorun altı halüsinasyon kabul edilecek
MAX_REFINEMENT_STEPS = 3       # Dinamik tamamlama için adım sınırı

# 5. Gramer Koruma ve Filtreleme Listeleri
BANNED_TOKENS = [
    ",", ".", "!", "?", ";", "-", "_",  # Noktalama işaretleri
    "and", "the", "a", "an", "of", "with", "in", "on", "at", "by", "for", "to", "from", # Bağlaçlar
    "is", "are", "was", "were", "be", "been", "being" # Yardımcı fiiller
]

print(f"[CONFIG] Dinamik Dizin Kilitlendi: {BASE_DIR}")
print("[CONFIG] Yasakli Token Filtreleri Başariyla Yüklendi.")