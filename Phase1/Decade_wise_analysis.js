
var aoi = ee.FeatureCollection("projects/quantumquotients/assets/Extended_CMA");

var startDate = '2023-08-14'
var endDate = '2023-10-16'

function applyScaleFactors(image) {
var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
return image.addBands(opticalBands, null, true)
          .addBands(thermalBands, null, true);
}


function maskL8sr(col) {
var cloudShadowBitMask = (1 << 3);
var cloudsBitMask = (1 << 5);
var qa = col.select('QA_PIXEL');
var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
             .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
return col.updateMask(mask);
}

var image = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
.filterDate(startDate, endDate)
.filterBounds(aoi)
.map(applyScaleFactors)
.map(maskL8sr)
.median()
.clip(aoi);

var visualization = {
bands: ['SR_B4', 'SR_B3', 'SR_B2'],
min: 0.0,
max: 0.3,
};

Map.addLayer(image, visualization, 'True Color (432)', false);

var ndvi  = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
Map.addLayer(ndvi, {min:-1, max:1, palette: ['blue', 'white', 'green']}, 'ndvi', false)

var ndvi_min = ee.Number(ndvi.reduceRegion({
reducer: ee.Reducer.min(),
geometry: aoi,
scale: 30,
maxPixels: 1e9
}).values().get(0))

var ndvi_max = ee.Number(ndvi.reduceRegion({
reducer: ee.Reducer.max(),
geometry: aoi,
scale: 30,
maxPixels: 1e9
}).values().get(0))


var fv = (ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min))).pow(ee.Number(2))
      .rename('FV')


var em = fv.multiply(ee.Number(0.004)).add(ee.Number(0.986)).rename('EM')

var thermal = image.select('ST_B10').rename('thermal')


var lst = thermal.expression(
    '(tb / (1 + ((11.5 * (tb / 14380)) * log(em)))) - 273.15',
    {
        'tb': thermal.select('thermal'), 
        'em': em                       
    }
).rename('LST');

var lst_mean = ee.Number(lst.reduceRegion({
reducer: ee.Reducer.mean(),
geometry: aoi,
scale: 30,
maxPixels: 1e9
}).values().get(0))

var lst_minVis = lst_mean.subtract(5);
var lst_maxVis = lst_mean.add(5);

var lst_vis = {
  min: lst_minVis.getInfo(),
  max: lst_maxVis.getInfo(),
  palette: [
    '040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
    '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
    '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
    'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
    'ff0000', 'de0101', 'c21301', 'a71001', '911003'
  ]
};


Map.addLayer(lst, lst_vis, 'LST AOI')
Map.centerObject(aoi, 10)


var lst_std = ee.Number(lst.reduceRegion({
reducer: ee.Reducer.stdDev(),
geometry: aoi,
scale: 30,
maxPixels: 1e9
}).values().get(0))



print('Mean LST in AOI', lst_mean)
print('STD LST in AOI', lst_std)


var uhi = lst.subtract(lst_mean).divide(lst_std).rename('UHI')

var uhi_vis = {
min: -3,
max: 3,
palette:['313695', '74add1', 'fed976', 'feb24c', 'fd8d3c', 'fc4e2a', 'e31a1c',
'b10026']
}
Map.addLayer(uhi, uhi_vis, 'UHI AOI')


// ==========================
// Floating Label for LST
// ==========================

// Create a panel to display values
var infoPanel = ui.Panel({
  style: {
    position: 'top-center',
    padding: '8px',
    backgroundColor: 'rgba(255,255,255,0.8)'
  }
});
Map.add(infoPanel);

// Add default text
var label = ui.Label('Click on the map to get LST (°C)');
infoPanel.add(label);

// Click event on the map
Map.onClick(function(coords) {
  var point = ee.Geometry.Point(coords.lon, coords.lat);
  
  var value = lst.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 30,
    maxPixels: 1e9
  });
  
  value.evaluate(function(val) {
    if (val && val.LST !== null) {
      label.setValue(
        ' LST at (' + coords.lon.toFixed(4) + ', ' + coords.lat.toFixed(4) + '): ' +
        val.LST.toFixed(2) + ' °C'
      );
    } else {
      label.setValue('No LST data at this location.');
    }
  });
});