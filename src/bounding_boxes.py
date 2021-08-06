from src.ETL.ee_boundingbox import BoundingBox

bounding_boxes = {
    "Kenya": BoundingBox(min_lon=33.501, max_lon=42.283, min_lat=-5.202, max_lat=6.002),
    "Kenya_Busia": BoundingBox(
        min_lon=33.88389587402344,
        min_lat=-0.04119872691853491,
        max_lon=34.44007873535156,
        max_lat=0.7779454563313616,
    ),
    "Mali_USAID_ZOIS_upper": BoundingBox(
        min_lon=-5.197335399999872,
        max_lon=-1.9996060219999094,
        min_lat=13.1527367220001,
        max_lat=17.11443622600001,
    ),
    "Mali_USAID_ZOIS_lower": BoundingBox(
        min_lon=-8.34855327799994,
        max_lon=-4.661010148999935,
        min_lat=10.36852747000006,
        max_lat=12.583134347000112,
    ),
    "Malawi_North": BoundingBox(
        min_lon=-8.34855327799994,
        max_lon=-4.661010148999935,
        min_lat=10.36852747000006,
        max_lat=12.583134347000112,
    ),
    "Malawi_South": BoundingBox(min_lon=34.211, max_lon=35.772, min_lat=-17.07, max_lat=-14.636),
    "Rwanda": BoundingBox(min_lon=28.841, max_lon=30.909, min_lat=-2.854, max_lat=-1.034),
    "RwandaSake": BoundingBox(min_lon=30.377, max_lon=30.404, min_lat=-2.251, max_lat=-2.226),
    "Togo": BoundingBox(
        min_lon=-0.1501, max_lon=1.7779296875, min_lat=6.08940429687, max_lat=11.115625
    ),
    "Uganda": BoundingBox(
        min_lon=29.5794661801, max_lon=35.03599, min_lat=-1.44332244223, max_lat=4.24988494736
    ),
}
