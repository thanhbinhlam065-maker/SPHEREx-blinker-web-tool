An end-to-end astronomical image-processing and kinematic tracking web application built for the **NASA Space Apps Challenge**. The pipeline ingests multi-epoch FITS/MEF cutouts from **NASA SPHEREx** and **unWISE / Legacy Survey**, homogenizes heterogeneous exposures onto a common astrometric and photometric frame, and detects moving celestial bodies (high-proper-motion stars, brown dwarfs, and solar system objects) with quantitative completeness validation.

🔗 **Live Demo:** `(https://spherex-blinker-web-tool-by-faster-than-light-group.streamlit.app)`  
🎥 **Video Demonstration:** `(https://drive.google.com/file/d/16N5wpTXAjdbEPcUSZu9uOczOdTaUNw4w/view?usp=drive_link)`

> **⚠️ Service Status Note (IRSA TAP Outage Fallback):**  
> Because NASA's IRSA TAP service experienced downtime during the submission window, this repository comes **pre-loaded with a real multi-epoch SPHEREx MEF dataset** (`sample_data/[your_file_name].fits`) so judges can immediately run the full analysis pipeline out of the box. Live querying from **unWISE / Legacy Survey** and custom **FITS/MEF file uploads** are also fully supported.

---

## ✨ Key Scientific & Engineering Features

1. **Multi-Epoch Data Ingestion & Quality Filtering**
   - Supports Multi-Extension FITS (MEF) stacks, individual FITS files, and direct sky-coordinate queries.
   - Parses SPHEREx bitmask planes (`FLAGS`) to mask detector artifacts (`TRANSIENT`, `SNOWBALL`, `HALO`, `GHOST`, `SATURATED`, `DEAD`, `HOT`) while preserving real astrophysical detections (`SOURCE`).

2. **Astrometric, PSF & Photometric Homogenization (`Process_fits_data.py`)**
   - **WCS Reprojection:** Aligns multi-epoch cutouts onto a common pixel grid using `reproject` (`reproject_interp` with flux conservation).
   - **PSF Matching:** Measures seeing/FWHM across epochs and homogenizes spatial resolution via Gaussian kernel convolution so stationary background stars subtract cleanly.
   - **Flux Scaling:** Normalizes background levels and linearizes zero-point flux scales across exposures before differencing.

3. **Interactive Multi-Epoch Blinker & Difference Imaging**
   - Frame-by-frame **Blinker** with adjustable playback speed, stretch modes (`asinh`, `log`, `linear`, `zscale`), and difference/residual overlays to visually inspect proper motion and transient residuals.

4. **Source Detection, Kinematic Linking & Injection-Recovery**
   - Detects candidate sources per epoch using `photutils` (`DAOStarFinder`) and links linear trajectories across MJD timestamps (`tracking.py`) to measure proper motion ($\mu_\alpha \cos\delta$, $\mu_\delta$ in $\text{mas/yr}$ or $\text{arcsec/yr}$).
   - **Completeness Testing:** Injects synthetic moving sources across varying magnitudes and velocities to compute recovery efficiency curves and validate pipeline sensitivity.

5. **Bilingual Interface (English / Tiếng Việt)**
   - Built with Streamlit for rapid interactive exploration, exporting results as CSV catalogs and publication-ready diagnostic plots.

---

## 🖼️ Interface & Science Preview

| Multi-Epoch Blinker & Alignment | Proper Motion Fit & Injection-Recovery |

---

## 📂 Repository Structure

```text
├── app.py                  # Main Streamlit web application (bilingual UI & workflow)
├── data_io.py              # FITS/MEF loader, header/WCS parser, and bitmask handler
├── Process_fits_data.py    # WCS reprojection, PSF matching, and flux scaling engine
├── SPHEREx.py              # NASA IRSA TAP & SPHEREx cutout retrieval module
├── tracking.py             # Multi-epoch source detection & linear tracklet linking
├── viz.py                  # Astronomical visualization, blinker frames & diagnostic plots
├── sample_data/            # Pre-loaded real SPHEREx MEF dataset for instant offline demo
└── requirements.txt        # Python dependencies
