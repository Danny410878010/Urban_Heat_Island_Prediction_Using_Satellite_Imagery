# Urban Heat Island (UHI) Prediction Using Satellite Imagery & XGBoost
The project for UMD Info challenge.
This repository presents our approach to predicting Urban Heat Island (UHI) intensity using satellite image data and machine learning. By combining remote sensing spectral bands, engineered features, and advanced modeling techniques, we achieved a highly accurate prediction model with an **R² of 0.94** on testing data.

---

## What We Did

### 1. **Feature Extraction**
We collected satellite image bands (B01–B12) in `.tiff` format to capture diverse spectral information relevant to land surface characteristics.

### 2. **Feature Engineering**
To enhance the predictive power of the dataset, we derived several domain-informed indices:
- **NDVI** – Normalized Difference Vegetation Index: `(B08 - B04) / (B08 + B04)`
- **NDBI** – Normalized Difference Built-up Index: `(B11 - B08) / (B11 + B08)`
- **Moisture Content**: `B11 / B12`
- **Soil Ratio**: `B06 / B07`
- **Vegetation Health**: `B08 / B02`

### 3. **Model Selection**
We evaluated multiple models:
- Linear Regression
- Random Forest
- Neural Network
- **XGBoost (Best performance)**

XGBoost outperformed others in capturing the **non-linear relationships** within the data, with a dramatic R² improvement over baselines.

### 4. **Parameter Tuning**
Our best XGBoost configuration:
- `n_estimators=1000`
- `learning_rate=0.005`
- `max_depth=20`
- `reg_lambda=0.001` (L2 regularization)

### 5. **Buffered Region Averaging**
Instead of predicting on single pixels, we averaged surrounding pixel values within a **2000-meter buffer**.  
**Critical Improvement**: R² increased from **0.53 to 0.93**  
This reflects the environmental influence on local UHI and better mimics real-world heat distribution.

### 6. **Feature Selection & Optimization**
By removing highly correlated and redundant features:
- **Reduced multicollinearity**
- Slight R² boost: **0.93 → 0.9414**
- Cut training time: **2 min → 1 min**

Final feature set:
```python
['B01', 'B02', 'B05', 'B06', 'B08', 'B8A', 'B11', 'B12', 
 'NDVI', 'NDBI', 'Moisture Content', 'Soil Ratio', 'Vegetation Health']

