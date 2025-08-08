from pathlib import Path
from PIL import Image, ImageDraw

examples_dir = Path("examples")
examples_dir.mkdir(exist_ok=True)

def create_invoice_image(filename, company_name, bill_to, items):
    W, H = 1600, 1000
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    title = "INVOICE"
    lines = [
        f"Company: {company_name}",
        "Address: 123 Market St, San Francisco, CA 94103",
        "Invoice #: INV-2025-081",
        "Date: 2025-08-07",
        "",
        f"Bill To: {bill_to}",
        "Item                 Qty    Unit Price    Amount",
    ]

    # Add item lines
    for item_name, qty, unit_price in items:
        amount = qty * unit_price
        lines.append(f"{item_name:<20} {qty:<5} ${unit_price:<10.2f} ${amount:<10.2f}")

    lines.extend([
        "",
        f"Subtotal:                              ${sum(q*up for _,q,up in items):.2f}",
        "Tax (8.5%):                            $ 61.46",
        "Total:                                 $784.46",
    ])

    # Draw
    d.text((W//2 - 120, 40), title, fill="black")
    y = 120
    for line in lines:
        d.text((80, y), line, fill="black")
        y += 48

    img.save(examples_dir / filename, dpi=(300, 300))
    print(f"✅ Created {filename}")

# Create multiple clean invoice images
create_invoice_image(
    "invoice_clean_acme.png",
    "Acme Robotics, Inc.",
    "Grindr LLC",
    [("GPU Cluster Hours", 120, 4.20),
     ("Storage (TB-Mo)", 10, 12.00),
     ("Support (Premium)", 1, 99.00)]
)

create_invoice_image(
    "invoice_clean_contoso.png",
    "Contoso Ltd.",
    "OpenAI Inc.",
    [("LLM Training Hours", 50, 20.00),
     ("Cloud Storage", 5, 15.00),
     ("Consulting", 2, 150.00)]
)
