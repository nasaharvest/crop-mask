var malawi_binary = ee.Image("projects/ee-aasareansah/assets/malawi_binary"),
    Sentinel2 = ee.Image("projects/ee-aasareansah/assets/Sent_Sept_Comp_Malawi"),
    Malawi_EPA = ee.FeatureCollection("projects/ee-aasareansah/assets/NASA_Harvest/Malawi_EPAs"),
    Malawi_TA = ee.FeatureCollection("projects/ee-aasareansah/assets/NASA_Harvest/Malawi_TAs"),
    Admin = ee.FeatureCollection("projects/ee-aasareansah/assets/NASA_Harvest/Malawi_admin2");


//Rename b1 as Wrong value 
var upd_name = malawi_binary.select('b1').rename('Wrong value');

var leftMap = ui.Map();
var layer1 = ui.Map.GeometryLayer([], 'Geometry Layer', 'red');
var drawingTools1 = leftMap.drawingTools();
drawingTools1.layers().add(layer1);

var rightMap = ui.Map();
var layer2 = ui.Map.Layer(ee.FeatureCollection([]), {'color': 'red'}, 'Points');

// Link two layers together
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

//Set map center using this long lat and zoom level
rightMap.setCenter(33.78, -12.27, 11);

// Make the default basemap the satellite layer
rightMap.setOptions('satellite');

//Specify the point as the only drawing tool 
drawingTools1.setDrawModes(['point']);

//Assign specific palettes to the cropland map
var palettes = require('users/gena/packages:palettes');
var palette = palettes.cmocean.Speed[7];

// Use this empty image for paint().
var empty = ee.Image().byte();

var Admin_color = empty.paint({
featureCollection: Admin,
color: 1,
width: 2
});

var EPA_color = empty.paint({
featureCollection: Malawi_EPA,
color: 1,
width: 2
});

var TA_color = empty.paint({
featureCollection: Malawi_TA,
color: 1,
width: 2
});

//Display the cropland map on the left panel
var vis_crop = {min: 0, max: 1.0, palette: palette.slice(0,-2)};
leftMap.addLayer(malawi_binary, vis_crop, 'Malawi cropland');
leftMap.addLayer(Admin_color, {palette: 'CE33FF'}, 'Malawi_Admin');
leftMap.addLayer(EPA_color, {palette: '0096FF'}, 'Malawi_EPA', false);
leftMap.addLayer(TA_color, {palette: 'black'}, 'Malawi_TA', false);

//Visualize the Sentinel 2 high resolution image on the right panel
var vis = {min: 0.0, max: 0.3, bands: ['B4', 'B3', 'B2']};
rightMap.addLayer(Sentinel2, vis, 'Sentinel-2');
rightMap.add(layer2);
rightMap.addLayer(Admin_color, {palette: 'CE33FF'}, 'Malawi_Admin');
rightMap.addLayer(EPA_color, {palette: '0096FF'}, 'Malawi_EPA', false);
rightMap.addLayer(TA_color, {palette: 'black'}, 'Malawi_TA', false);

// Link the two panels
var linker = ui.Map.Linker([leftMap, rightMap]);

//Ceate a label for the data export
var downloadUrl = ui.Label('', {
  stretch: 'horizontal',
  textAlign: 'left',
  fontWeight: 'bold',
  textDecoration: 'underline',
  fontSize: '10',
  height: '15px',
});

//Create a button to download all the data points as a feature collection
var button = ui.Button('Get link for downloading points',function() {
  var firstLayer = drawingTools1.layers().get(0);
  var fc = firstLayer.getEeObject();
  var sampled = upd_name.sample({
    region: fc,
    geometries: true,
  });
  //print(sampled);
  sampled = sampled.map(function(feature) {
    var point = feature.geometry().coordinates();
    var long = point.get(0);
    var lat = point.get(1);
    return feature.set({
      longitude: long,
      latitude: lat,
    });
  });
  
  //Generate a URL for the points and export as CSV
  //button.setLabel('Saved!');
  downloadUrl.setValue('');
  button.setDisabled(true);
  sampled.getDownloadURL({
    filename: 'Malawi_new points',
    callback: function(url) {
      downloadUrl.setValue(url);
      downloadUrl.setUrl(url);
      button.setDisabled(false);
    }
  });
  
});

// Create the split panels
var splitPanel = ui.SplitPanel({
  firstPanel: leftMap,
  secondPanel: rightMap,
  orientation: 'horizontal',
  wipe: false,
  style: {stretch: 'both'}
});


// Add captions
var infoPanel1 = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px',
    stretch: "both",
  }
});

var infoPanel2 = ui.Panel({
  style: {
    position: 'bottom-right',
    padding: '8px 15px',
    stretch: "both",
  }
});

var info1 =ui.Label({
  value: "Malawi cropland map",
  style: {
    fontSize: '12px',
    margin: '0 0 3px 0',
    padding: '0'
  }
});

var info2 =ui.Label({
  value: "Malawi Sentinel 2 Image",
  style: {
    fontSize: '12px',
    margin: '0 0 3px 0',
    padding: '0'
  }
});

var title = ui.Panel({
  widgets: [
    ui.Label("", {
      stretch: 'horizontal',
    }),
    ui.Label("Corrective Labeling App - Malawi", {
      fontSize: '24px',
      fontWeight: 'bold',
      //fontWeight: '120',
    }),
    ui.Label("", {
      stretch: 'horizontal',
    }),
  ],
  layout: ui.Panel.Layout.flow('horizontal'),
});

//Adding a Legend
// set position of panel
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 10px',
    stretch: 'horizontal'
  }
});
 
// Create legend title
var legendTitle = ui.Label({
  value: 'Legend',
  style: {
    fontWeight: 'bold',
    position: 'bottom-left',
    fontSize: '15px',
    margin: '0 0 4px 0',
    padding: '0'
    }
});

// Creates and styles 1 row of the legend.
var makeRow = function(color, name) {
 
      // Create the label that is actually the colored box.
      var colorBox = ui.Label({
        style: {
          backgroundColor: '#' + color,
          // Use padding to give the box height and width.
          padding: '8px',
          margin: '0 0 4px 0'
        }
      });
 
      // Create the label filled with the description text.
      var description = ui.Label({
        value: name,
        style: {margin: '0 0 4px 6px'}
      });
 
      // return the panel
      return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
      });
};
legend.add(legendTitle); 
//  Palette with the colors
var palette =['197228', 'FFFDCE'];
 
// name of the legend
var names = ['Crop','Non Crop',];
 
//Add color and and names
for (var i = 0; i < 2; i++) {
  legend.add(makeRow(palette[i], names[i]));
  }  

//Add a description to the title
var description =ui.Label({
  value: 'This app allows users to add points to where the crop map is wrongly classified. Users can compare the cropland map on the left against the high resolution Sentinel 2 data on the right. The "Add a Marker" icon on the left panel can be selected to start creating the points. To delete a point, click the hand icon to select the specific point and hit the delete button. After picking all the points, users can click the "Get link for downloading points" button to generate a link that can be downloaded as a CSV file. A detailed instruction manual of the App can be found in the link below:' ,
  style: {
    stretch: 'horizontal',
    position: 'top-center',
    fontSize: '14px',
    textAlign: 'center',
    margin: '0 0 3px 0',
    padding: '0',
    color: '#3f3f3f',
  }
});

var manual_link = ui.Label({
  value: 'Detailed Instructions', 
  style: {stretch: 'horizontal',
    position: 'top-center',
    fontSize: '15px',
    //fontWeight: 'bold',
    textAlign: 'center',
    color: 'blue'}});
    manual_link.setUrl('https://bit.ly/3EPT57D');

var mainPanel = ui.Panel({
  widgets: [
    title,
    description, manual_link, splitPanel, button, downloadUrl 
  ],
  style: {
    width: '100%',
    height: '100%',
  }
});


infoPanel1.add(info1);
infoPanel2.add(info2);

rightMap.add(infoPanel2);
leftMap.add(infoPanel1);
leftMap.add(legend);

ui.root.widgets().reset([mainPanel]);
linker.get(0).setCenter(33.78, -12.27, 11);