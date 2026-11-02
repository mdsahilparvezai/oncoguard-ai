# ONCOGUARD AI - Data Card

## Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)

### Basic Information

| Property | Value |
|----------|-------|
| **Name** | Wisconsin Diagnostic Breast Cancer |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| **License** | CC BY 4.0 (Creative Commons Attribution 4.0 International) |
| **Citation** | Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. |
| **Original Publication** | Street, W.N., Wolberg, W.H. and Mangasarian, O.L. (1993). Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, vol. 1905, pp. 861-870. |

### Dataset Composition

| Property | Value |
|----------|-------|
| **Instances** | 569 |
| **Features** | 30 numerical + 1 ID + 1 target |
| **Missing Values** | 0 |
| **Duplicate Rows** | 0 |
| **Class Distribution** | Benign: 357 (62.7%), Malignant: 212 (37.3%) |

### Feature Description

The 30 features describe characteristics of cell nuclei extracted from digitized fine-needle aspirate (FNA) images of breast masses. For each of 10 nuclear characteristics, three statistics are computed:

| Base Feature | Description |
|--------------|-------------|
| **radius** | Mean distance from center to points on the perimeter |
| **texture** | Standard deviation of gray-scale values (texture) |
| **perimeter** | Perimeter of the nucleus |
| **area** | Area of the nucleus |
| **smoothness** | Local variation in radius lengths |
| **compactness** | Perimeter² / area - 1.0 |
| **concavity** | Severity of concave portions of the contour |
| **concave_points** | Number of concave portions of the contour |
| **symmetry** | Symmetry of the nucleus |
| **fractal_dimension** | "Coastline approximation" - 1 |

**Statistics computed for each:**
- `mean` — Mean value across nuclei
- `se` — Standard error (standard deviation / sqrt(n))
- `worst` — Mean of the three largest (worst) values

**Total: 10 × 3 = 30 features**

### Target Variable

| Original | Encoded | Meaning |
|----------|---------|---------|
| `M` | 1 | Malignant (positive class — cancer present) |
| `B` | 0 | Benign (negative class — no cancer) |

**Note:** Malignant is treated as the positive class (1) because in medical diagnosis, the "positive" outcome is the presence of disease. This is critical for interpreting sensitivity/recall (ability to detect cancer).

### Data Collection & Preprocessing (Original)

From the original publication:
1. **Image Acquisition:** Fine-needle aspirate (FNA) samples from breast masses
2. **Digitization:** Images digitized at 40x magnification
3. **Segmentation:** Automated nuclear boundary detection
4. **Feature Extraction:** 10 morphological features computed per nucleus
5. **Aggregation:** Mean, SE, and worst (mean of 3 largest) computed per case
6. **Diagnosis:** Confirmed by surgical biopsy (ground truth)

### Ethical Considerations

| Aspect | Details |
|--------|---------|
| **Informed Consent** | Original data collected under 1990s institutional review |
| **Patient Privacy** | No direct identifiers; features are abstracted measurements |
| **Demographics** | No age, race, gender, or clinical metadata available |
| **Bias Risks** | Single institution (University of Wisconsin); unknown population representativeness |
| **Temporal Scope** | Data from early 1990s; imaging and diagnostic standards have evolved |

### ONCOGUARD AI Usage

| Aspect | Implementation |
|--------|----------------|
| **Loading** | Downloaded from UCI URL, cached locally |
| **Validation** | Schema validation, checksum verification, quality checks |
| **Splitting** | Stratified 70% train / 10% validation / 20% test (seed=42) |
| **Preprocessing** | StandardScaler fit on training data only (leakage prevention) |
| **ID Handling** | Explicitly excluded from features |
| **Target Mapping** | M→1, B→0 (malignant=positive) |

### Limitations for Research

1. **Small Sample Size** — 569 cases limits model complexity and statistical confidence
2. **Single Center** — University of Wisconsin only; geographic/demographic bias likely
3. **No Clinical Metadata** — Age, menopausal status, family history, treatment outcomes unavailable
4. **Feature Level** — Abstracted nuclear features, not raw images or whole-slide data
5. **No External Validation** — Performance on independent cohorts completely unknown
6. **Class Imbalance** — 37% malignant requires balanced metrics and careful threshold selection
7. **Benchmark Scope** — Optimized for UCI benchmark performance, not clinical utility

### Version & Checksums

- **Raw File:** `data/raw/wdbc.data`
- **Processed Files:** `data/processed/*.parquet`
- **Checksum:** SHA256 computed on raw DataFrame for reproducibility verification

### Access & Licensing

- **UCI Repository:** Free for research/education with attribution
- **CC BY 4.0:** Requires attribution to UCI and original authors
- **Commercial Use:** Allowed with attribution (per CC BY 4.0)
- **Redistribution:** Allowed with attribution

### Recommended Citation for This Project

```
ONCOGUARD AI Research Platform. Wisconsin Diagnostic Breast Cancer dataset
from UCI Machine Learning Repository (Dua & Graff, 2019).
Original: Street et al., Nuclear feature extraction for breast tumor diagnosis, 1993.
```