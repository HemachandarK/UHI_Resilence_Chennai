import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import rioxarray as rxr
import rasterio
import mplcursors
import matplotlib.colors as mcolors

csv_path = "Chennai_UHI_TrainingData-8.csv" 
df = pd.read_csv(csv_path)
print("Original shape:", df.shape)

df = df[df['LST'] > 0].drop(columns=['.geo'], errors='ignore')
df = df[(df['LST'] > 270) & (df['LST'] < 330)] 
df['LST'] = df['LST'] - 273.15 
print("Cleaned shape:", df.shape)
print("Columns:", df.columns.tolist())

feature_cols = ['NDVI','NDBI','NDWI','Albedo','EVI','SAVI','UI','Elevation']
X = df[feature_cols].values.astype(np.float32)
y = df['LST'].values.astype(np.float32)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train size:", X_train.shape, "Val size:", X_val.shape, "Test size:", X_test.shape)

param_grid = {
    "max_depth": [6, 8, 10],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "n_estimators": [500, 1000]
}

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",  
    random_state=42
)

grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="r2",
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

print("\n Best parameters:", grid.best_params_)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n Final Test Set Results:")
print(f"  RMSE: {rmse:.2f} °C")
print(f"  MAE : {mae:.2f} °C")
print(f"  R²  : {r2:.3f}")

model_path = "xgb_model_extended.json"
best_model.save_model(model_path)
print("Best XGBoost model saved at:", model_path)

xgb.plot_importance(best_model, importance_type="gain")
plt.title("Feature Importance for UHI LST Prediction (Extended Features)")
plt.show()


ref_raster = rxr.open_rasterio("NDVI_10m.tif").squeeze() 

def load_and_align(path, ref):
    r = rxr.open_rasterio(path).squeeze()
    return r.rio.reproject_match(ref)

ndvi   = load_and_align("NDVI_10m.tif", ref_raster)
ndbi   = load_and_align("NDBI_10m-2.tif", ref_raster)
ndwi   = load_and_align("NDWI_10m-2.tif", ref_raster)
albedo = load_and_align("Albedo_10m-2.tif", ref_raster)
evi    = load_and_align("EVI_10m.tif", ref_raster)
savi   = load_and_align("SAVI_10m.tif", ref_raster)
ui     = load_and_align("UI_10m.tif", ref_raster)
elev   = load_and_align("Elevation_30m.tif", ref_raster)

stack = np.stack([
    ndvi.values, ndbi.values, ndwi.values, albedo.values,
    evi.values, savi.values, ui.values, elev.values
], axis=-1).astype(np.float32)
mask = np.any(np.isnan(stack), axis=-1)
X_raster = stack[~mask]

y_raster_pred = best_model.predict(X_raster)

pred_grid = np.full(mask.shape, np.nan, dtype=np.float32)
pred_grid[~mask] = y_raster_pred

with rasterio.open("NDVI_10m.tif") as src:  
    meta = src.meta.copy()

meta.update({"count": 1, "dtype": "float32"})

out_path = "LST_XGB_Extended.tif"
with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(pred_grid.astype("float32"), 1)

print("Final extended-feature LST map saved at:", out_path)


with rasterio.open("NDVI_10m.tif") as src: 
    bounds = src.bounds
    transform = src.transform

colors = ['#269db1',  
 '#fff705', 
 '#ff8b13', 
 '#ff0000', 
 '#911003']  
cmap = mcolors.ListedColormap(colors)

plt.figure(figsize=(10,8))
img = plt.imshow(pred_grid, cmap=cmap,
                 extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
plt.colorbar(label="LST (°C)", extend="both")
plt.title("Downscaled LST (10 m, Extended Features) - Chennai (March–May)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

cursor = mplcursors.cursor(img, hover=True)

@cursor.connect("add")
def on_hover(sel):
    x, y = sel.target  
    row, col = ~transform * (x, y)
    row, col = int(row), int(col)
    if 0 <= row < pred_grid.shape[0] and 0 <= col < pred_grid.shape[1]:
        value = pred_grid[row, col]
        sel.annotation.set_text(f"Lat: {y:.5f}\nLon: {x:.5f}\nLST: {value:.2f} °C")

plt.show()
