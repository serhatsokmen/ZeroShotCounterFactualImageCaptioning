import torch
import clip
from src import config

class CounterFactualEngine:
    def __init__(self, architecture):
        """
        models.py içinde tanımlanan MultiModalArchitecture sınıfını 
        kontrol motoruna bağlar.
        """
        self.arch = architecture

    def run_feedback_loop(self, raw_image):
        """
        Görseli alır, halüsinasyon denetimi yapar ve kısıtlı öz-bağlanımlı 
        cümle tamamlama mekanizması ile altyazıyı dinamik olarak optimize eder.
        """
        step_history = []
        
        # 1. Adım: Standart BLIP başlangıç altyazısını üret ve ilk CLIP skorunu al
        current_caption = self.arch.generate_initial_caption(raw_image)
        current_score = self.arch.calculate_clip_similarity(raw_image, current_caption)
        
        initial_caption = current_caption
        initial_score = current_score
        
        print(f"\n[ENGINE] Başlangıç Altyazısı: \"{current_caption}\"")
        print(f"[ENGINE] Başlangıç CLIP Skoru: {current_score:.4f}")

        # 2. Adım: İteratif Kısıtlı Geri Besleme Döngüsü (Feedback Loop)
        step = 0
        while current_score < config.HALUCINATION_THRESHOLD and step < config.MAX_REFINEMENT_STEPS:
            step += 1
            print(f"[DÖNGÜ ADIM {step}] Halüsinasyon Riski! Skor ({current_score:.4f}) < Eşik ({config.HALUCINATION_THRESHOLD})")
            print(f"[ENGINE] Kısıtlı Autoregressive Cümle Tamamlama Süreci Başlatıldı...")

            # Cümleyi kelimelerine ayırıyoruz
            words = current_caption.split()
            if len(words) < 3:
                print("[ENGINE] Cümle manipülasyon için çok kısa. Döngü kırılıyor.")
                break

            best_candidate = current_caption
            best_candidate_score = current_score

            # STRATEJİ: Cümlenin sonundaki riskli nesne alanlarını farklı derinliklerde maskeliyoruz.
            # Kesilen noktanın üzerine modelin yasaklı kelimeler olmadan yeni ve tam bir cümle kurmasını sağlıyoruz.
            for mask_depth in range(1, min(4, len(words) - 1)):
                prefix_words = words[:-mask_depth]
                prefix_text = " ".join(prefix_words)
                
                # Modeller katmanında mühürlenen donanım seviyesindeki filtreleri tetikleyerek
                # tam ve anlamlı alternatif tamamlama cümle dizilerini topluyoruz
                candidates = self.arch.generate_constrained_continuation(
                    raw_image, 
                    prefix_text, 
                    max_alternatives=5
                )
                
                # Türetilen tam cümlelerin her birini anlık olarak CLIP uzayında skorluyoruz
                for candidate in candidates:
                    cand_score = self.arch.calculate_clip_similarity(raw_image, candidate)
                    
                    # Eğer üretilen tam cümle, görselle mevcut cümleden daha yüksek hizalanıyorsa güncelle
                    if cand_score > best_candidate_score:
                        best_candidate_score = cand_score
                        best_candidate = candidate
                        print(f"[ANLAMLI ADAY] \"{candidate}\" -> Yeni CLIP Skoru: {cand_score:.4f}")

            # İterasyon sonu kontrolü ve cümlenin güncellenmesi
            if best_candidate_score > current_score:
                print(f"[DÖNGÜ ADIM {step}] Altyazı Başarıyla Güncellendi -> \"{best_candidate}\"")
                print(f"[DÖNGÜ ADIM {step}] Yeni En İyi CLIP Skoru: {best_candidate_score:.4f}")
                current_caption = best_candidate
                current_score = best_candidate_score
                step_history.append((current_caption, current_score))
            else:
                # Dil modeli kısıtlamalar nedeniyle skoru geliştirecek kelime bulamadıysa pes etme
                # Güvenli prefix alanını jenerik olarak temiz bırakıp döngüyü sonlandır
                print(f"[ENGINE] Dil modeli filtreleri skoru daha fazla geliştiremedi. Güvenli taban daraltılıyor...")
                fallback_caption = " ".join(words[:-2])
                fallback_score = self.arch.calculate_clip_similarity(raw_image, fallback_caption)
                if fallback_score > current_score:
                    current_caption = fallback_caption
                    current_score = fallback_score
                    step_history.append((current_caption, current_score))
                break

        if current_score >= config.HALUCINATION_THRESHOLD:
            print(f"[ENGINE] Başarılı! Nihai altyazı gramer korunarak güvenli bölgeye ulaştı. Nihai Skor: {current_score:.4f}")
        else:
            print(f"ℹ[ENGINE] Döngü sınırına ulaşıldı. Skor optimize edildi ancak eşiğin altında kaldı.")

        return initial_caption, initial_score, current_caption, current_score, step_history