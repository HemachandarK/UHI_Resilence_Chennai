import pandas as pd
import numpy as np
import rasterio
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# === 1. Load Ground Truth from Excel ===
excel_path = "ground_truth_LST.xlsx"   # downloaded sheet
df_gt = pd.read_excel(excel_path)

# Expected columns: latitude, longitude, measured_LST
lat_col, lon_col, lst_col = "Latitude", "Longitude", "LST_Avg"

# === 2. Load Predicted LST GeoTIFF ===
tif_path = "LST_XGB_Extended.tif"
dataset = rasterio.open(tif_path)

# === 3. Extract Predicted LST at Ground Points ===
predicted_lst = []
for _, row in df_gt.iterrows():
    lon, lat = row[lon_col], row[lat_col]
    try:
        row_pix, col_pix = dataset.index(lon, lat)
        value = dataset.read(1)[row_pix, col_pix]
        if np.isnan(value):
            value = None
    except:
        value = None
    predicted_lst.append(value)

df_gt["Predicted_LST"] = predicted_lst

# === 4. Remove Missing Predictions ===
df_val = df_gt.dropna(subset=["Predicted_LST"])

# === 5. Compute Metrics ===
y_true = df_val[lst_col].values
y_pred = df_val["Predicted_LST"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

# === Save final CSV with predicted & actual LST ===
output_csv = "Ground_Validation_with_Predicted.csv"

# Only keep rows where prediction exists
df_val = df_gt.dropna(subset=["Predicted_LST"])

# Save to CSV
df_val.to_csv(output_csv, index=False)

print(f"✅ Final CSV saved to '{output_csv}'")
print("\nValidation Metrics:")
print(f"  RMSE: {rmse:.2f} °C")
print(f"  MAE : {mae:.2f} °C")
print(f"  R²  : {r2:.3f}")


# === 6. Scatter Plot ===
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, c="blue", alpha=0.6, edgecolors="k")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
plt.xlabel("Ground Measured LST (°C)")
plt.ylabel("Predicted LST (°C)")
plt.title("Ground Truth vs Predicted LST Validation")
plt.grid(True)
plt.show()

# === 7. Save Results to Excel ===
# === Save LST values as CSV ===

