from src.ETL.ee_boundingbox import BoundingBox

bounding_boxes = {
    "Ethiopia_Tigray": BoundingBox(min_lon=36.45, max_lon=40.00, min_lat=12.25, max_lat=14.895),
    "Kenya": BoundingBox(min_lon=33.501, max_lon=42.283, min_lat=-5.202, max_lat=6.002),
    "Kenya_Busia": BoundingBox(
        min_lon=33.88389587402344,
        min_lat=-0.04119872691853491,
        max_lon=34.44007873535156,
        max_lat=0.7779454563313616,
    ),
    "Mali": BoundingBox(
        min_lon=-12.170750,
        max_lon=4.270210,
        min_lat=10.096361,
        max_lat=24.974574,
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
    "Uganda_surrounding_2": BoundingBox(
        min_lon=27.5794661801, max_lon=37.03599, min_lat=-3.44332244223, max_lat=6.24988494736
    ),
    "Uganda_surrounding_5": BoundingBox(
        min_lon=24.5794661801, max_lon=40.03599, min_lat=-6.44332244223, max_lat=9.24988494736
    ),
    "Uganda_surrounding_10": BoundingBox(
        min_lon=19.5794661801, max_lon=45.03599, min_lat=-11.44332244223, max_lat=14.24988494736
    ),
    "Uganda_surrounding_20": BoundingBox(
        min_lon=9.5794661801, max_lon=55.03599, min_lat=-21.44332244223, max_lat=24.24988494736
    ),
    "Global": BoundingBox(min_lon=-180, max_lon=180, min_lat=-90, max_lat=90),
}
