import os 
from PIL import Image
import matplotlib.pyplot as plt 
from src import config

def load_raw_image(image_path):
    """Verilen dosya yolundaki gorseli PIL formatinda yukler."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[UTILS] HATA: {image_path} konumunda gorsel bulunamadi!")
    return Image.open(image_path).convert("RGB")


def visualize_results(image_path, initial_caption, initial_score, refined_caption, refined_score, step_history):
    """
    Canli 'before/after' grafik penceresini olusturur ve outputs klasorune kaydeder.
    Sol tarafta gorsel, sag tarafta ise altyazi karsilastirmalari yer alir.
    """
    # 1. Grafik penceresinin boyutlarini ayarlama 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Zero-Shot Counter-Factual Image Captioning - Hallucination Mitigation", fontsize=16, fontweight="bold")

    # Orijinal Test Gorseli 
    raw_image = Image.open(image_path)
    axes[0].imshow(raw_image)
    axes[0].axis("off")  # Piksel koordinat cizgilerini gizlemek icin
    axes[0].set_title(f"Giriş Görseli: {os.path.basename(image_path)}", fontsize=12, fontweight="semibold")

    # Akademik Sonuc ve Skor Karsilastirmasi 
    axes[1].axis("off")  # Sag taraf metin tablosu olarak kullanilir
    axes[1].set_title("Mimarinin Denetim ve Düzeltme Raporu", fontsize=12, fontweight="semibold")

    y_pos = 0.85

    # Standart Model Ciktisi
    axes[1].text(0.05, y_pos, "Standart Model (BLIP) Baslangic Ciktisi:", fontsize=11, fontweight="bold", color="crimson")
    axes[1].text(0.05, y_pos - 0.06, f"Altyazi: \"{initial_caption}\"", fontsize=11, style="italic")
    axes[1].text(0.05, y_pos - 0.12, f"CLIP Kosinus Benzerligi: {initial_score:.4f}", fontsize=11, color="crimson")

    axes[1].text(0.05, y_pos - 0.20, "--------------------------------------------------------------------------------", fontsize=10, color="gray")

    axes[1].text(0.05, y_pos - 0.28, "Feedback Loop Nihai Ciktisi:", fontsize=11, fontweight="bold", color="forestgreen")
    axes[1].text(0.05, y_pos - 0.34, f"Altyazi: \"{refined_caption}\"", fontsize=11, style="italic")
    axes[1].text(0.05, y_pos - 0.40, f"CLIP Kosinus Benzerligi: {refined_score:.4f}", fontsize=11, color="forestgreen")

    # Durum Degerlendirmesi
    axes[1].text(0.05, y_pos - 0.52, "Sistem Durum Notu:", fontsize=11, fontweight="bold", color="darkblue")
    if initial_score < refined_score:
        improvement = ((refined_score - initial_score) / initial_score) * 100
        status_text = f"Halüsinasyon Basariyla Engellendi!\nAnlamsal Dogruluk %{improvement:.1f} oraninda arttirildi.\nToplam Iterasyon Adimi: {len(step_history)}"
        axes[1].text(0.05, y_pos - 0.62, status_text, fontsize=11, color="darkgreen", fontweight="medium")
    else:
        axes[1].text(0.05, y_pos - 0.62, "Ilk altyazi zaten guvenli esigin uzerindeydi.\nDuzeltmeye ihtiyac duyulmadi.", fontsize=11, color="darkblue")
    
    
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"result_{os.path.basename(image_path)}"
    save_path = os.path.join(output_dir, output_filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[UTILS] Grafik basariyla olusturuldu ve kaydedildi: {save_path}")

    plt.show()