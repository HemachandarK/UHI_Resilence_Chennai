/***************************************
 * STEP 0: REGION OF INTEREST
 ***************************************/
var roi = cmda.geometry();
Map.centerObject(roi, 9);
Map.addLayer(roi, { color: "red" }, "CMDA");

/***************************************
 * STEP 1: BUILDING FOOTPRINT BINARY (λp)
 * Dataset: Open Buildings v3
 ***************************************/
var buildings = ee
  .FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
  .filterBounds(roi);

// Rasterize building polygons
var building_binary = ee
  .Image()
  .byte()
  .paint({
    featureCollection: buildings,
    color: 1,
  })
  .clip(roi)
  .rename("building");

Map.addLayer(building_binary, { min: 0, max: 1 }, "Building Footprint Binary");

/***************************************
 * STEP 2: BUILDING HEIGHT RASTER (H̄, σH)
 * Dataset: Open Buildings 2.5D Temporal
 ***************************************/
var building_height = ee
  .ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
  .filterBounds(roi)
  .filter(ee.Filter.calendarRange(2023, 2023, "year"))
  .mosaic()
  .select("building_height")
  .clip(roi)
  .rename("height");

Map.addLayer(building_height, { min: 0, max: 50 }, "Building Height");

/***************************************
 * STEP 3: LAND COVER (REFERENCE)
 * Dataset: ESA WorldCover 2020
 ***************************************/
var landcover = ee.Image("ESA/WorldCover/v100/2020").clip(roi);

Map.addLayer(landcover, {}, "ESA WorldCover");

/***************************************
 * STEP 4: OPEN SPACE BINARY (OSR)
 * Grass = 40, Bare = 60, Water = 80
 ***************************************/
var open_space = landcover
  .eq(40)
  .or(landcover.eq(60))
  .or(landcover.eq(80))
  .rename("open");

Map.addLayer(open_space, { min: 0, max: 1 }, "Open Space Binary");

/***************************************
 * STEP 5: STUDY AREA MASK
 ***************************************/
var mask = ee.Image.constant(1).clip(roi).rename("mask");

/***************************************
 * STEP 6: EXPORT ALL RASTERS AS GeoTIFF
 ***************************************/

// 1️⃣ Building footprint binary (λp)
Export.image.toDrive({
  image: building_binary,
  description: "CMDA_building_footprint_binary",
  region: roi,
  scale: 4,
  crs: "EPSG:32644",
  maxPixels: 1e13,
});

// 2️⃣ Building height raster (H̄, σH)
Export.image.toDrive({
  image: building_height,
  description: "CMDA_building_height",
  region: roi,
  scale: 4,
  crs: "EPSG:32644",
  maxPixels: 1e13,
});

// 3️⃣ Open space binary (OSR)
Export.image.toDrive({
  image: open_space,
  description: "CMDA_open_space_binary",
  region: roi,
  scale: 10,
  crs: "EPSG:32644",
  maxPixels: 1e13,
});

// 4️⃣ Landcover reference
Export.image.toDrive({
  image: landcover,
  description: "CMDA_landcover",
  region: roi,
  scale: 10,
  crs: "EPSG:32644",
  maxPixels: 1e13,
});

// 5️⃣ Boundary mask
Export.image.toDrive({
  image: mask,
  description: "CMDA_mask",
  region: roi,
  scale: 10,
  crs: "EPSG:32644",
  maxPixels: 1e13,
});
