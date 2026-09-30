import os 
from src import config
from src.models import MultiModalArchitecture
from src.engine import CounterFactualEngine
from src import utils

def main():
    print("==============================================================================")
    print("      ZERO-SHOT COUNTER-FACTUAL IMAGE CAPTIONING SYSTEM STARTING              ")
    print("==============================================================================")

    # Mimariyi ve kontrol motorunu baslat
    architecture = MultiModalArchitecture()
    engine = CounterFactualEngine(architecture)

    # Dinamik gorsel klasoru
    image_dir = config.IMAGE_DIR

    if not os.path.exists(image_dir):
        print(f"[MAIN] HATA: {image_dir} klasoru bulunamadi!")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"[MAIN] HATA: {image_dir} klasoru icinde hic gorsel bulunamadi!")
        return 
    
    print(f"[MAIN] Toplam {len(image_files)} adet test gorseli algilandi. Islem basliyor...\n")

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"\n" + "="*60)
        print(f"[ISLEM] Gorsel Analiz Ediliyor: {img_file}")
        print("="*60)

        try:
            # Gorseli yukle
            raw_image = utils.load_raw_image(img_path)

            # Geri besleme dongusunu calistir
            initial_cap, initial_score, refined_cap, refined_score, history = engine.run_feedback_loop(raw_image)

            # Sonuclari gorsellestir ve kaydet
            print(f"[MAIN] Gorsel Sonuc Raporu Olusturuluyor...")
            utils.visualize_results(img_path, initial_cap, initial_score, refined_cap, refined_score, history)
        
        except Exception as e:
            print(f"[MAIN] HATA: {img_file} islenirken bir sorun olustu! Detay: {str(e)}")
            continue
    
    print("\n==============================================================================")
    print("SUCCESS: Tum test gorselleri basariyla islendi ve raporlar kaydedildi.")
    print("==============================================================================")


if __name__ == "__main__":
    main()