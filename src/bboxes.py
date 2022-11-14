from openmapflow.bbox import BBox

bboxes = {
    "East_Africa": BBox(min_lat=-11.829, max_lat=6.003, min_lon=28.430, max_lon=42.284),
    "Ethiopia_Bure_Jimma": BBox(min_lat=7, max_lat=11, min_lon=34, max_lon=38),
    "Ethiopia_Tigray": BBox(min_lat=12.25, max_lat=14.910, min_lon=36.45, max_lon=40.00),
    "Global": BBox(min_lat=-90, max_lat=90, min_lon=-180, max_lon=180),
    "Kenya": BBox(min_lat=-5.202, max_lat=6.002, min_lon=33.501, max_lon=42.283),
    "Malawi": BBox(min_lat=-17.135, max_lat=-9.230, min_lon=32.546, max_lon=36.224),
    "Mali_lower": BBox(min_lat=10.368, max_lat=12.584, min_lon=-8.348, max_lon=-4.662),
    "Mali_upper": BBox(min_lat=13.152, max_lat=17.134, min_lon=-5.197, max_lon=-1.998),
    "Namibia_North": BBox(min_lat=-19.484, max_lat=-17.379, min_lon=14.151, max_lon=25.045),
    "Rwanda": BBox(min_lat=-3.035, max_lat=-0.760, min_lon=28.430, max_lon=31.013),
    "Sudan_Blue_Nile": BBox(min_lat=9.454, max_lat=12.734, min_lon=33.082, max_lon=35.149),
    "Togo": BBox(min_lat=6.089, max_lat=11.134, min_lon=-0.1501, max_lon=1.778),
    "Uganda": BBox(min_lat=-1.63, max_lat=4.30, min_lon=29.12, max_lon=35.18),
    "Tanzania": BBox(min_lat=-13.794, max_lat=0.801, min_lon=28.082, max_lon=41.331),
    "Zambia": BBox(min_lat=-18.334, max_lat=-7.536, min_lon=21.379, max_lon=33.991),
}
