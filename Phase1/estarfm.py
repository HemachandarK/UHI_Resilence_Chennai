from utils import read_raster, writeimage
import math
import numpy as np
from osgeo import gdal
import os
import datetime
import yaml
import statsmodels.api as sm
from scipy.stats import f
from sklearn.linear_model import LinearRegression

try:
    import idlwrap
except Exception:
    class _IDLWrapFallback:
        @staticmethod
        def indgen(*dims):
            if len(dims) == 1:
                n = int(dims[0])
                return np.arange(n, dtype=np.int64)
            elif len(dims) == 2:
                r, c = map(int, dims)
                arr = np.arange(r*c, dtype=np.int64).reshape(r, c)
                return arr
        @staticmethod
        def intarr(*dims):
            if len(dims) == 1:
                n = int(dims[0])
                return np.zeros(n, dtype=np.int64)
            elif len(dims) == 2:
                r, c = map(int, dims)
                return np.zeros((r, c), dtype=np.int64)
            else:
                return np.zeros(tuple(map(int, dims)), dtype=np.int64)
    idlwrap = _IDLWrapFallback()

try:
    np.int
except AttributeError:
    np.int = int
np.seterr(divide='ignore', invalid='ignore')

config_path = "/content/drive/MyDrive/ESTARFM/config.yaml"
with open(config_path, "r") as f:
    param = yaml.safe_load(f)

w = param['w']
num_class = param['num_class']
DN_min = param['DN_min']
DN_max = param['DN_max']
background = param['background']
patch_long = param['patch_long']

temp_file = param['temp_file']
os.makedirs(temp_file, exist_ok=True)

path1 = param['path1']
path2 = param['path2']
path3 = param['path3']
path4 = param['path4']
path5 = param['path5']

def compute_ndvi(red_band, nir_band):
    return (nir_band - red_band) / (nir_band + red_band + 1e-10)

def save_single_band(array, ref_path, out_path):
    ds = gdal.Open(ref_path)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(out_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(array)
    out_ds.FlushCache()
    out_ds = None
    ds = None

landsat_ds1 = gdal.Open(path1)
landsat_ndvi1 = compute_ndvi(
    landsat_ds1.GetRasterBand(3).ReadAsArray().astype(float),
    landsat_ds1.GetRasterBand(4).ReadAsArray().astype(float)
)
landsat_ds1 = None
landsat_ndvi1_path = os.path.splitext(path1)[0] + "_NDVI.tif"
save_single_band(landsat_ndvi1, path1, landsat_ndvi1_path)

landsat_ds2 = gdal.Open(path3)
landsat_ndvi2 = compute_ndvi(
    landsat_ds2.GetRasterBand(3).ReadAsArray().astype(float),
    landsat_ds2.GetRasterBand(4).ReadAsArray().astype(float)
)
landsat_ds2 = None
landsat_ndvi2_path = os.path.splitext(path3)[0] + "_NDVI.tif"
save_single_band(landsat_ndvi2, path3, landsat_ndvi2_path)

modis_ds1 = gdal.Open(path2)
if modis_ds1.RasterCount == 1:
    modis_ndvi1_path = path2
else:
    red_m = modis_ds1.GetRasterBand(1).ReadAsArray().astype(float)
    nir_m = modis_ds1.GetRasterBand(2).ReadAsArray().astype(float)
    modis_ndvi1 = compute_ndvi(red_m, nir_m)
    modis_ndvi1_path = os.path.splitext(path2)[0] + "_NDVI.tif"
    save_single_band(modis_ndvi1, path2, modis_ndvi1_path)
modis_ds1 = None

modis_ds2 = gdal.Open(path4)
if modis_ds2.RasterCount == 1:
    modis_ndvi2_path = path4
else:
    red_m = modis_ds2.GetRasterBand(1).ReadAsArray().astype(float)
    nir_m = modis_ds2.GetRasterBand(2).ReadAsArray().astype(float)
    modis_ndvi2 = compute_ndvi(red_m, nir_m)
    modis_ndvi2_path = os.path.splitext(path4)[0] + "_NDVI.tif"
    save_single_band(modis_ndvi2, path4, modis_ndvi2_path)
modis_ds2 = None

modis_ds3 = gdal.Open(path5)
if modis_ds3.RasterCount == 1:
    modis_ndvi3_path = path5
else:
    red_m = modis_ds3.GetRasterBand(1).ReadAsArray().astype(float)
    nir_m = modis_ds3.GetRasterBand(2).ReadAsArray().astype(float)
    modis_ndvi3 = compute_ndvi(red_m, nir_m)
    modis_ndvi3_path = os.path.splitext(path5)[0] + "_NDVI.tif"
    save_single_band(modis_ndvi3, path5, modis_ndvi3_path)
modis_ds3 = None

path1 = landsat_ndvi1_path
path2 = modis_ndvi1_path
path3 = landsat_ndvi2_path
path4 = modis_ndvi2_path
path5 = modis_ndvi3_path

def ensure_match_grid(ref_path, in_path, out_path):
    ref = gdal.Open(ref_path)
    if ref is None:
        raise RuntimeError(f"Cannot open reference raster: {ref_path}")
    gt = ref.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    pixel_w = gt[1]
    pixel_h = gt[5]
    xmax = xmin + pixel_w * ref.RasterXSize
    ymin = ymax + pixel_h * ref.RasterYSize

    warp_opts = gdal.WarpOptions(
        format="GTiff",
        xRes=abs(pixel_w),
        yRes=abs(pixel_w) if pixel_h == 0 else abs(pixel_h),
        resampleAlg="bilinear",
        dstSRS=ref.GetProjection(),
        outputBounds=(xmin, ymin, xmax, ymax),
        targetAlignedPixels=True,
        multithread=True
    )
    res = gdal.Warp(out_path, in_path, options=warp_opts)
    if res is None:
        raise RuntimeError(f"gdal.Warp failed: {in_path} -> {out_path}")
    res = None

print("Ensuring grids match Landsat (path1) ...")
aligned_path2 = os.path.join(temp_file, "aligned_C1.tif")
aligned_path4 = os.path.join(temp_file, "aligned_C2.tif")
aligned_path5 = os.path.join(temp_file, "aligned_C0.tif")

ensure_match_grid(path1, path2, aligned_path2)
ensure_match_grid(path1, path4, aligned_path4)
ensure_match_grid(path1, path5, aligned_path5)

path2 = aligned_path2
path4 = aligned_path4
path5 = aligned_path5

print("Reading input images and initializing variables...")

orig_nl, orig_ns, temp_data = read_raster(path1)
nb = temp_data.shape[0]

suffix = '.tif' if path1.lower().endswith('.tif') else ('.dat' if path1.lower().endswith('.dat') else '')

print(f"Image dimensions: {orig_nl} x {orig_ns}, Bands: {nb}")
print(f"File format: {suffix}")

n_nl = int(math.ceil(float(orig_nl) / patch_long))
n_ns = int(math.ceil(float(orig_ns) / patch_long))
print(f"Number of blocks: {n_nl} x {n_ns} = {n_nl * n_ns}")

ind_patch = np.zeros((n_nl * n_ns, 4), dtype=int)
patch_id = 0
for i_ns in range(n_ns):
    for i_nl in range(n_nl):
        col1 = i_ns * patch_long
        col2 = min(col1 + patch_long - 1, orig_ns - 1)
        row1 = i_nl * patch_long
        row2 = min(row1 + patch_long - 1, orig_nl - 1)
        ind_patch[patch_id, :] = [col1, col2, row1, row2]
        patch_id += 1

def split_image_into_patches(input_path, output_prefix):
    nl, ns, data = read_raster(input_path)
    if nl != orig_nl or ns != orig_ns:
        raise RuntimeError(f"Size mismatch: {input_path} is {nl}x{ns}, expected {orig_nl}x{orig_ns}")

    for i in range(n_nl * n_ns):
        col1, col2, row1, row2 = ind_patch[i, :]
        height = row2 - row1 + 1
        width  = col2 - col1 + 1
        if height <= 0 or width <= 0:
            print(f" Skipping invalid patch {i+1}: size {height}x{width}")
            continue

        patch_data = data[:, row1:row2+1, col1:col2+1]
        if patch_data.shape[1] == 0 or patch_data.shape[2] == 0:
            print(f" Skipping empty patch {i+1} at rows {row1}:{row2}, cols {col1}:{col2}")
            continue

        output_path = os.path.join(temp_file, f"{output_prefix}{i+1}{suffix}")
        writeimage(patch_data, output_path, input_path)

print("Splitting images into patches...")
split_image_into_patches(path1, "temp_F1")
split_image_into_patches(path2, "temp_C1")
split_image_into_patches(path3, "temp_F2")
split_image_into_patches(path4, "temp_C2")
split_image_into_patches(path5, "temp_C0")
print("Image splitting completed.")

starttime = datetime.datetime.now()
print('there are total', n_nl*n_ns, 'blocks')

for isub in range(0, n_nl * n_ns):
    FileName = os.path.join(temp_file, f"temp_F1{isub + 1}{suffix}")
    nl, ns, fine1 = read_raster(FileName)

    FileName = os.path.join(temp_file, f"temp_C1{isub + 1}{suffix}")
    _, _, coarse1 = read_raster(FileName)

    FileName = os.path.join(temp_file, f"temp_F2{isub + 1}{suffix}")
    _, _, fine2 = read_raster(FileName)

    FileName = os.path.join(temp_file, f"temp_C2{isub + 1}{suffix}")
    _, _, coarse2 = read_raster(FileName)

    FileName = os.path.join(temp_file, f"temp_C0{isub + 1}{suffix}")
    _, _, coarse0 = read_raster(FileName)

    fine0 = np.zeros((nb, nl, ns)).astype(float)

    row_index = np.zeros((nl, ns), dtype=int)
    for ii in range(nl):
        row_index[ii, :] = ii
    col_index = np.zeros((nl, ns), dtype=int)
    for ii in range(ns):
        col_index[:, ii] = ii

    uncertain = (DN_max*0.002) * np.sqrt(2)

    similar_th = np.zeros((2, nb)).astype(float)
    for iband in range(nb):
        similar_th[0, iband] = np.std(fine1[iband, :, :] * 2.0 / num_class)
        similar_th[1, iband] = np.std(fine2[iband, :, :] * 2.0 / num_class)

    yy, xx = np.meshgrid(np.arange(-w, w+1), np.arange(-w, w+1))
    D_D_all = 1.0 + np.sqrt(xx**2 + yy**2) / float(w)
    D_D_all = D_D_all.flatten()

    valid_index = np.zeros((nl, ns), dtype=int)
    ind_valid = np.where(
        (fine1[0, :, :] != background) &
        (fine2[0, :, :] != background) &
        (coarse1[0, :, :] != background) &
        (coarse2[0, :, :] != background) &
        (coarse0[0, :, :] != background)
    )
    if len(ind_valid[0]) > 0:
        valid_index[ind_valid] = 1

    for j in range(nl):
        for i in range(ns):
            if valid_index[j, i] != 1:
                continue

            ai = int(max(0, i - w))
            bi = int(min(ns - 1, i + w))
            aj = int(max(0, j - w))
            bj = int(min(nl - 1, j + w))

            ind_wind_valid = np.where((valid_index[aj:bj+1, ai:bi+1]).ravel() == 1)
            position_cand = idlwrap.intarr((bi-ai+1)*(bj-aj+1)) + 1
            row_wind = row_index[aj:bj+1, ai:bi+1]
            col_wind = col_index[aj:bj + 1, ai:bi + 1]

            for ipair in [0, 1]:
                for iband in range(nb):
                    cand_band = idlwrap.intarr((bi-ai+1)*(bj-aj+1))
                    if ipair == 0:
                        S_S = np.abs(fine1[iband, aj:bj+1, ai:bi+1] - fine1[iband, j, i])
                    else:
                        S_S = np.abs(fine2[iband, aj:bj + 1, ai:bi + 1] - fine2[iband, j, i])
                    ind_cand = np.where(S_S.ravel() < similar_th[ipair, iband])
                    cand_band[ind_cand] = 1
                    position_cand = position_cand * cand_band

            indcand = np.where((position_cand != 0) & ((valid_index[aj:bj+1, ai:bi+1]).ravel() == 1))
            number_cand = len(indcand[0])

            if number_cand > 5:
                S_D_cand = np.zeros(number_cand).astype(float)
                x_cand = (col_wind.ravel())[indcand]
                y_cand = (row_wind.ravel())[indcand]
                finecand = np.zeros((nb*2, number_cand)).astype(float)
                coarsecand = np.zeros((nb*2, number_cand)).astype(float)

                for ib in range(nb):
                    finecand[ib, :]    = (fine1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                    finecand[ib+nb, :] = (fine2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                    coarsecand[ib, :]    = (coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                    coarsecand[ib+nb, :] = (coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]

                if nb == 1:
                    S_D_cand = 1.0 - 0.5*(np.abs((finecand[0, :]-coarsecand[0, :]) / (finecand[0, :]+coarsecand[0, :])) +
                                          np.abs((finecand[1, :]-coarsecand[1, :]) / (finecand[1, :]+coarsecand[1, :])))
                else:
                    sdx = np.std(finecand, axis=0, ddof=1)
                    sdy = np.std(coarsecand, axis=0, ddof=1)
                    meanx = np.mean(finecand, axis=0)
                    meany = np.mean(coarsecand, axis=0)

                    x_meanx = finecand - meanx
                    y_meany = coarsecand - meany
                    S_D_cand = nb*2.0*np.mean(x_meanx*y_meany, axis=0) / (sdx*sdy) / (nb*2.0-1)

                ind_nan = np.where(~np.isfinite(S_D_cand))
                if len(ind_nan[0]) > 0:
                    S_D_cand[ind_nan] = 0.5

                if (bi-ai+1)*(bj-aj+1) < (w*2.0+1)*(w*2.0+1):
                    D_D_cand = 1.0 + np.sqrt((i-x_cand)**2+(j-y_cand)**2) / w
                else:
                    D_D_cand = D_D_all[indcand]

                C_D = (1.0-S_D_cand) * D_D_cand + 1e-7
                weight = (1.0/C_D)/np.sum(1.0/C_D)

                for ib in range(nb):
                    fine_cand = np.hstack(((fine1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand],
                                           (fine2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]))
                    coarse_cand = np.hstack(((coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand],
                                             (coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]))
                    coarse_change = np.abs(np.mean((coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]) -
                                           np.mean((coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]))

                    if coarse_change >= DN_max*0.02:
                        X = coarse_cand.reshape(-1, 1)
                        Y = fine_cand.reshape(-1, 1)
                        XX = sm.add_constant(X)
                        model = sm.OLS(Y, XX).fit()
                        regress_result = model.params
                        sig = model.f_pvalue
                        if sig <= 0.05 and 0 < regress_result[1] <= 5:
                            V_cand = regress_result[1]
                        else:
                            V_cand = 1.0
                    else:
                        V_cand = 1.0

                    difc_pair1 = np.abs(np.mean((coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid]) -
                                        np.mean((coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid])) + 1e-10
                    difc_pair2 = np.abs(np.mean((coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid]) -
                                        np.mean((coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid])) + 1e-10
                    T_weight1 = (1.0/difc_pair1) / ((1.0/difc_pair1)+(1.0/difc_pair2))
                    T_weight2 = (1.0/difc_pair2) / ((1.0/difc_pair1)+(1.0/difc_pair2))

                    coase0_cand = (coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                    coase1_cand = (coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]
                    coase2_cand = (coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand]

                    fine01 = fine1[ib, j, i] + np.sum(weight * V_cand * (coase0_cand-coase1_cand))
                    fine02 = fine2[ib, j, i] + np.sum(weight * V_cand * (coase0_cand-coase2_cand))
                    fine0[ib, j, i] = T_weight1 * fine01 + T_weight2 * fine02

                    if fine0[ib, j, i] <= DN_min or fine0[ib, j, i] >= DN_max:
                        fine01 = np.sum(weight*(fine1[ib, aj:bj+1, ai:bi+1]).ravel()[indcand])
                        fine02 = np.sum(weight*(fine2[ib, aj:bj+1, ai:bi+1]).ravel()[indcand])
                        fine0[ib, j, i] = T_weight1 * fine01 + T_weight2 * fine02

            else:
                for ib in range(nb):
                    difc_pair1 = np.mean((coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid]) - \
                                 np.mean((coarse1[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid]) + 1e-10
                    difc_pair2 = np.mean((coarse0[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid]) - \
                                 np.mean((coarse2[ib, aj:bj+1, ai:bi+1]).ravel()[ind_wind_valid]) + 1e-10
                    difc_pair1_a = np.abs(difc_pair1)
                    difc_pair2_a = np.abs(difc_pair2)
                    T_weight1 = (1.0/difc_pair1_a) / ((1.0/difc_pair1_a)+(1.0/difc_pair2_a))
                    T_weight2 = (1.0/difc_pair2_a) / ((1.0/difc_pair1_a)+(1.0/difc_pair2_a))
                    fine0[ib, j, i] = T_weight1 * (fine1[ib, j, i] + difc_pair1) + \
                                      T_weight2 * (fine2[ib, j, i] + difc_pair2)

    print('finish ', str(isub + 1), 'block')
    tempoutname1 = os.path.join(temp_file, 'temp_blended')
    Out_Name = f"{tempoutname1}{isub + 1}{suffix}"
    writeimage(fine0, Out_Name, path1)

print("Mosaicking blended patches...")

datalist = []
xOffset_list = []
yOffset_list = []

minx_list = []
miny_list = []
for isub in range(n_ns * n_nl):
    col1, col2, row1, row2 = ind_patch[isub, :]
    minx_list.append(col1)
    miny_list.append(row1)

minX = min(minx_list)
minY = min(miny_list)

for isub in range(n_ns * n_nl):
    out_name = os.path.join(temp_file, f"temp_blended{isub+1}{suffix}")
    if not os.path.exists(out_name):
        print(f" Missing blended patch, skipping: {out_name}")
        continue
    datalist.append(out_name)
    col1, col2, row1, row2 = ind_patch[isub, :]
    xOffset_list.append(int(col1 - minX))
    yOffset_list.append(int(row1 - minY))

in_ds = gdal.Open(path1)
out_final = os.path.splitext(path5)[0] + "_ESTARFM" + suffix

driver = gdal.GetDriverByName("GTiff") if suffix == '.tif' else gdal.GetDriverByName("ENVI")
dataset = driver.Create(out_final, orig_ns, orig_nl, nb, gdal.GDT_Float32)

for i, data in enumerate(datalist):
    nl, ns, datavalue = read_raster(data)
    for j in range(nb):
        dd = datavalue[j, :, :]
        dataset.GetRasterBand(j + 1).WriteArray(dd, xOffset_list[i], yOffset_list[i])

dataset.SetGeoTransform(in_ds.GetGeoTransform())
dataset.SetProjection(in_ds.GetProjection())

print("ESTARFM processing completed successfully!")
print(f"Output saved to: {out_final}")

endtime = datetime.datetime.now()
print(f"Total processing time: {endtime - starttime}")