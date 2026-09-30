import torch 
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip 
from src import config 

class MultiModalArchitecture:
    def __init__(self):
        # 1. Donanım kontrolü
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MODELS] Calistirma Donanimi Algilandi: {self.device.type.upper()}")

        # 2. BLIP modelinin yüklenmesi (Image Captioner)
        print("[MODELS] BLIP modeli ve islemcisi yukleniyor...")
        self.blip_processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_NAME)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(config.BLIP_MODEL_NAME).to(self.device)

        # 3. CLIP modelinin yüklenmesi (Halüsinasyon Doğrulayıcı)
        print("[MODELS] OpenAI CLIP modeli yukleniyor...")
        self.clip_model, self.clip_preprocess = clip.load(config.CLIP_MODEL_NAME, device=self.device)

        # 4. Derin Öğrenme Standart mod ayarları (ağırlıkları dondurma)
        self.blip_model.eval()
        self.clip_model.eval()
        
        # Yasaklı kelimeleri BLIP'in tokenizer ID'lerine dönüştürme işlemi
        self.banned_ids = [self.blip_processor.tokenizer.encode(w, add_special_tokens=False) for w in config.BANNED_TOKENS]
        self.banned_ids = [ids for ids in self.banned_ids if len(ids) > 0]
        
        print("[MODELS] Tum derin ogrenme katmanlari EVAL modunda. Yasakli token ID'leri muhurlendi.")

    def generate_initial_caption(self, raw_image):
        """Görseli alıp BLIP ile standart başlangıç altyazısını üretir."""
        inputs = self.blip_processor(images=raw_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.blip_model.generate(**inputs, max_new_tokens=40)
        caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    def calculate_clip_similarity(self, raw_image, text_prompt):
        """Görsel ile metin promptu arasındaki kosinüs benzerliğini hesaplar."""
        image_input = self.clip_preprocess(raw_image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([text_prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()
        
        return similarity

    def generate_constrained_continuation(self, raw_image, prefix_text, max_alternatives=5):
        """
        Güvenli prefix metnini alır. Yasaklı kelimeler (virgül, bağlaçlar vb.)
        kullanılmasını engelleyerek cümlenin geri kalanını anlamlı şekilde tamamlar.
        """
        inputs = self.blip_processor(images=raw_image, text=prefix_text, return_tensors="pt").to(self.device)
        
        candidates = []
        with torch.no_grad():
            # Nucleus Sampling (top_p) ve kısıtlı arama parametreleri
            outputs = self.blip_model.generate(
                **inputs,
                max_new_tokens=15,
                num_return_sequences=max_alternatives,
                do_sample=True,
                top_k=50,
                top_p=0.92,
                bad_words_ids=self.banned_ids,
                repetition_penalty=1.2
            )
            
            for output in outputs:
                decoded_sentence = self.blip_processor.decode(output, skip_special_tokens=True).strip()
                if decoded_sentence and decoded_sentence not in candidates:
                    candidates.append(decoded_sentence)
                    
        return candidates