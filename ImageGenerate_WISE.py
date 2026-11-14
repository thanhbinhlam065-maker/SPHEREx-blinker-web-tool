import os
import io
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt

# 1. Danh sách mục tiêu kiểm thử
targets = {
    "Barnards_Star": {"ra": "17h 57m 48.5s", "dec": "+04d 41m 36s", "desc": "Di chuyển rất nhanh"},
    "Epsilon_Indi_B": {"ra": "22h 04m 10.5s", "dec": "-56d 46m 58s", "desc": "Hệ sao lùn nâu"},
    "WISE_0855": {"ra": "08h 55m 10.831s", "dec": "-07d 14m 42.53s", "desc": "Lùn nâu cực nhanh"},
    "RW_Aurigae": {"ra": "05h 07m 49.57s", "dec": "+30d 24m 05.2s", "desc": "Sao biến quang"},
    "R_Draconis": {"ra": "16h 32m 40.2s", "dec": "+66d 45m 17.6s", "desc": "Sao biến quang"}
}

output_dir = "ImageGenerate_WISE"
os.makedirs(output_dir, exist_ok=True)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# Danh sách chính xác tất cả các lớp dữ liệu unWISE trên Legacy Survey
# Gồm giai đoạn khảo sát AllWISE gốc (2010) và 11 năm NEOWISE (2014 - 2024)
unwise_layers = {
    "2010": "unwise-allwise",  # Khảo sát gốc AllWISE
    "2014": "unwise-neo1",  # NEOWISE Year 1 (12/2013 - 12/2014)
    "2015": "unwise-neo2",  # NEOWISE Year 2
    "2016": "unwise-neo3",  # NEOWISE Year 3
    "2017": "unwise-neo4",  # NEOWISE Year 4
    "2018": "unwise-neo5",  # NEOWISE Year 5
    "2019": "unwise-neo6",  # NEOWISE Year 6
    "2020": "unwise-neo7",  # NEOWISE Year 7
    "2021": "unwise-neo8",  # NEOWISE Year 8
    "2022": "unwise-neo9",  # NEOWISE Year 9
    "2023": "unwise-neo10",  # NEOWISE Year 10
    "2024": "unwise-neo11"  # NEOWISE Year 11 (Data mới nhất hiện tại)
}

print("--- FIRST SATGE: QUÉT TÌM MỐC THỜI GIAN CÓ DATA ---")
valid_epochs = {}

for year_str, layer in unwise_layers.items():
    print(f"\nĐang kiểm tra mốc {year_str} (layer {layer})...")
    epoch_cache = {}

    for name, info in targets.items():
        coord = SkyCoord(f"{info['ra']} {info['dec']}", frame='icrs')
        fits_url = (
            f"https://www.legacysurvey.org/viewer/fits-cutout?"
            f"ra={coord.ra.deg:.6f}&dec={coord.dec.deg:.6f}&layer={layer}"
            f"&pixscale=2.75&size=200"
        )

        try:
            response = requests.get(fits_url, headers=headers, timeout=20)
            if response.status_code == 200 and response.content.startswith(b'SIMPLE'):
                epoch_cache[name] = response.content
                print(f"  + {name}: Tải thành công")
            else:
                print(f"  - {name}: Không có data")
        except Exception as e:
            print(f"  - {name}: Lỗi ({e})")

    if epoch_cache:
        print(f"=> Cập nhật mốc {year_str}: {len(epoch_cache)}/{len(targets)} thiên thể có ảnh.")
        valid_epochs[year_str] = epoch_cache
    else:
        print(f"=> Trống hoàn toàn. Bỏ qua mốc {year_str}.")

if not valid_epochs:
    print("Không tìm thấy data nào. Hủy quá trình.")
    exit()

print(f"\n=> Sẽ xử lý {len(valid_epochs)} mốc thời gian: {', '.join(valid_epochs.keys())}")
print("\n--- SECOND STAGE: XỬ LÝ VÀ XUẤT ẢNH ---")

for year, target_data in valid_epochs.items():
    for name, fits_bytes in target_data.items():
        print(f"Xuất file cho {name} - Năm {year}")

        # 1. Lưu file FITS_DATA gốc
        fits_filename = os.path.join(output_dir, f"{name}_{year}_WISE.fits")
        with open(fits_filename, 'wb') as f:
            f.write(fits_bytes)

        # Trích xuất và chuẩn hóa mảng dữ liệu W1
        with fits.open(io.BytesIO(fits_bytes)) as hdul:
            data = hdul[0].data
            w1_data = data[0] if data.ndim == 3 else data
            norm = ImageNormalize(w1_data, interval=ZScaleInterval())

            # 2. PHIÊN BẢN ANNOTATED (Có trục & Legend)
            annotated_png = os.path.join(output_dir, f"{name}_{year}_annotated.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(w1_data, cmap='magma', origin='lower', norm=norm)
            plt.colorbar(label='Pixel Intensity (ZScale)')
            plt.title(f"{name} - {year} (W1 3.4µm)")
            plt.xlabel("RA (pixels)")
            plt.ylabel("DEC (pixels)")
            plt.savefig(annotated_png, bbox_inches='tight', dpi=150)
            plt.close()

            # 3. PHIÊN BẢN CLEAN PNG (Không viền/trục, dùng làm GIF / Blinking)
            clean_png = os.path.join(output_dir, f"{name}_{year}_clean.png")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(w1_data, cmap='magma', origin='lower', norm=norm)
            ax.axis('off')
            fig.savefig(clean_png, bbox_inches='tight', pad_inches=0, dpi=150, transparent=True)
            plt.close(fig)

print("\nHoàn tất! Toàn bộ file đã sẵn sàng trong thư mục 'ImageGenerate_WISE'")