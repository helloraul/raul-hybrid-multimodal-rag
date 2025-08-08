from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import os

image_path = "examples/invoice_image.png"

print(f"📂 Image exists: {os.path.exists(image_path)}")
if not os.path.exists(image_path):
    print(f" File not found: {image_path}")
    raise SystemExit(1)

def preprocess(p, contrast=2.0, sharpen_times=1, threshold=None, invert=False, upscale=2):
    img = Image.open(p).convert("L")  # grayscale
    if invert:
        img = ImageOps.invert(img)
    for _ in range(max(0, sharpen_times)):
        img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # 🔼 Upscale before any binarization; Tesseract prefers ~300 DPI+
    if upscale and upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.BICUBIC)

    # Optional binarization (off by default)
    if threshold is not None:
        img = img.point(lambda v: 255 if v > threshold else 0, mode="1")

    out = os.path.splitext(p)[0] + "_processed.png"
    img.save(out)
    return out

try:
    # ✅ Minimal-work defaults: stronger contrast, more sharpen, NO threshold
    processed = preprocess(image_path, contrast=2.2, sharpen_times=2, threshold=None, upscale=2)

    print(f"🔍 Running OCR on: {processed}")
    # ✅ LSTM-only, sparse text, fake DPI to 300
    config = "--oem 1 --psm 11 -c user_defined_dpi=300"
    text = pytesseract.image_to_string(Image.open(processed), lang="eng", config=config)

    cleaned = text.strip()
    print("📝 OCR Result:\n" + (cleaned if cleaned else "❌ No text detected in image."))

    # Quick confidence peek
    try:
        data = pytesseract.image_to_data(
            Image.open(processed), lang="eng", config=config, output_type=pytesseract.Output.DICT
        )
        words = [(t, float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and c not in ("-1", -1)]
        if words:
            avg = sum(c for _, c in words) / len(words)
            top = sorted(words, key=lambda x: x[1], reverse=True)[:8]
            print(f"\n📊 Confidence: avg={avg:.1f}")
            for w, c in top:
                print(f"  - '{w}' ({c:.1f})")
        else:
            print("\n📊 Confidence: no valid tokens")
    except Exception:
        pass

    # 🔎 Optional: quick PSM sweep to see what works best (uncomment to use)
    try:
        print("\n🔄 Quick PSM sweep (preview):")
        for psm in [3, 4, 6, 7, 11, 12, 13]:
            cfg = f"--oem 1 --psm {psm} -c user_defined_dpi=300"
            txt = pytesseract.image_to_string(Image.open(processed), lang="eng", config=cfg).strip()
            snippet = (txt[:120] + "…") if len(txt) > 120 else txt
            print(f"\n=== PSM {psm} ===\n{snippet if snippet else '(no text)'}")
    except Exception:
        pass

except pytesseract.TesseractNotFoundError:
    print("❌ Tesseract not found on PATH.")
except Exception as e:
    print(f"❌ OCR failed with error: {e}")
