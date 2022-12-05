var cropland_binary = ee.Image("projects/ee-aasareansah/assets/malawi_binary");
var country = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Malawi'));

//Function to mask clouds using the Sentinel-2 QA band
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate('2020-09-01', '2021-09-01')
                  .filterBounds(country)
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
                  .map(maskS2clouds);

var Sentinel2 = s2.median().clip(country);
    
var upd_name = cropland_binary.select('b1').rename('Wrong value');

var leftMap = ui.Map();
var layer1 = ui.Map.GeometryLayer([], 'Geometry Layer', 'red');
var drawingTools1 = leftMap.drawingTools();
drawingTools1.layers().add(layer1);

var rightMap = ui.Map();
rightMap.setOptions('satellite');
var layer2 = ui.Map.Layer(ee.FeatureCollection([]), {'color': 'red'}, 'Points');

drawingTools1.onDraw(function(geometry, layer) {
  if (layer === layer1) {
    layer2.setEeObject(layer.getEeObject());
  }
});
drawingTools1.onEdit(ui.util.debounce(function(geometry, layer) {
  if (layer === layer1) {
    layer2.setEeObject(layer.getEeObject());
  }
}, 100));
drawingTools1.onErase(function(geometry, layer) {
  if (layer === layer1) {
    layer2.setEeObject(layer.getEeObject());
  }
});
drawingTools1.setDrawModes(['point']);

var empty = ee.Image().byte();
var country_color = empty.paint({featureCollection: country, color: 1, width: 2});

var palettes = require('users/gena/packages:palettes');
var palette = palettes.cmocean.Speed[7];

var vis_crop = {min: 0, max: 1.0, palette: palette.slice(0,-2)};
leftMap.addLayer(country_color, {palette:'black'}, 'border');
leftMap.addLayer(cropland_binary, vis_crop, 'Cropland binary');

var vis = {min: 0.0, max: 0.3, bands: ['B4', 'B3', 'B2']};
rightMap.addLayer(Sentinel2, vis, 'Sentinel-2');
rightMap.add(layer2);
rightMap.addLayer(country_color, {palette:'black'}, 'border');

var linker = ui.Map.Linker([leftMap, rightMap]);

var downloadUrl = ui.Label('', {stretch: 'horizontal', textAlign: 'left', fontWeight: 'bold', textDecoration: 'underline', fontSize: '10', height: '15px'});

var button = ui.Button('Get link for downloading points',function() {
  var firstLayer = drawingTools1.layers().get(0);
  var fc = firstLayer.getEeObject();
  var sampled = upd_name.sample({region: fc, geometries: true});
  
  sampled = sampled.map(function(feature) {
    var point = feature.geometry().coordinates();
    return feature.set({
        longitude: point.get(0), 
        latitude: point.get(1)});
  });
  
  downloadUrl.setValue('');
  button.setDisabled(true);
  sampled.getDownloadURL({
    filename: 'New points',
    callback: function(url) {
      downloadUrl.setValue(url);
      downloadUrl.setUrl(url);
      button.setDisabled(false)}});
});

var splitPanel = ui.SplitPanel({firstPanel: leftMap, secondPanel: rightMap, orientation: 'horizontal', wipe: false, style: {stretch: 'both'}});

var infoPanel1 = ui.Panel({style: {position: 'bottom-left', padding: '8px 15px', stretch: "both"}});
var infoPanel2 = ui.Panel({style: {position: 'bottom-right', padding: '8px 15px', stretch: "both"}});

var infoStyle = {fontSize: '12px', margin: '0 0 3px 0', padding: '0'};
var info1 = ui.Label({value: "Cropland map", style: infoStyle});
var info2 = ui.Label({value: "Sentinel 2 Image", style: infoStyle});
infoPanel1.add(info1);
infoPanel2.add(info2);

var title = ui.Panel({
  widgets: [
    ui.Label("", {stretch: 'horizontal'}),
    ui.Label("Corrective Labeling App", {fontSize: '24px', fontWeight: 'bold'}),
    ui.Label("", {stretch: 'horizontal'})],
  layout: ui.Panel.Layout.flow('horizontal')});

var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px 10px', stretch: 'horizontal'}});
var legendTitle = ui.Label({value: 'Legend', style: {fontWeight: 'bold', position: 'bottom-left', fontSize: '15px', margin: '0 0 4px 0', padding: '0'}});

var makeRow = function(color, name) {
    var colorBoxStyle = {backgroundColor: '#' + color, padding: '8px', margin: '0 0 4px 0'};
    var colorBox = ui.Label({style: colorBoxStyle});
    var description = ui.Label({value: name, style: {margin: '0 0 4px 6px'}});
    return ui.Panel({widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')});
  };
  legend.add(makeRow('197228', 'Crop'));
  legend.add(makeRow('FFFDCE', 'Non Crop'));

var description =ui.Label({
  value: 'This app allows users to add points to where the crop map is wrongly classified. Users can compare the cropland map on the left against the high resolution Sentinel 2 data on the right. The "Add a Marker" icon on the left panel can be selected to start creating the points. To delete a point, click the hand icon to select the specific point and hit the delete button. After picking all the points, users can click the "Get link for downloading points" button to generate a link that can be downloaded as a CSV file. A detailed instruction manual of the App can be found in the link below:' ,
  style: {stretch: 'horizontal', position: 'top-center', fontSize: '14px', textAlign: 'center', margin: '0 0 3px 0', padding: '0', color: '#3f3f3f'}});

var manual_link = ui.Label({
  value: 'Detailed Instructions', 
  style: {stretch: 'horizontal', position: 'top-center', fontSize: '15px', textAlign: 'center', color: 'blue'}});
  manual_link.setUrl('https://bit.ly/3EPT57D');

var mainPanel = ui.Panel({widgets: [title, description, manual_link, splitPanel, button, downloadUrl], style: {width: '100%', height: '100%'}});

rightMap.add(infoPanel2);
leftMap.add(infoPanel1);
leftMap.add(legend);

ui.root.widgets().reset([mainPanel]);
linker.get(0).centerObject(country, 8);