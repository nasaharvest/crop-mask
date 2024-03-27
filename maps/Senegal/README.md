# Senegal

## Preliminary Information
<!-- References -->
[1]: https://www.cia.gov/the-world-factbook/countries/senegal/#geography
[2]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5451287/
[3]: https://www.britannica.com/place/Senegal/Land
[4]: https://www.fao.org/giews/countrybrief/country.jsp?code=SEN&lang=ES
[5]: https://ipad.fas.usda.gov/countrysummary/default.aspx?id=SG
[6]: https://www.fao.org/faostat/en/#data/RL


### Admin Zones
<a title="Amitchell125, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Senegal,_administrative_divisions_in_colour_2.svg"><img width="512" alt="Senegal, administrative divisions in colour 2" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Senegal%2C_administrative_divisions_in_colour_2.svg/512px-Senegal%2C_administrative_divisions_in_colour_2.svg.png"></a>

### Geography
**Area** <sup>[1]</sup>
- Total: 196,722 sq km
- Land: 192,530 sq km
- Water: 4,192 sq km

**Landscape**
- Majority of landscape is rolling sandy plains which rise to foothills in the southeast
- Northern border is formed by the Senegal River

**Terrestrial Ecoregions** <sup>[2]</sup>
1. Guinean forest-savanna mosaic
2. Sahelian Acacia savanna
3. West Sudanian savanna, and
4. Guinean mangroves

**Climate** <sup>[1]</sup>
- Rainy season: May to November
- Dry season: December to April

**Soils** <sup>[3]</sup>
- In northwest soils are highly favorable for peanut cultivation
- Center and South of the country have a layer of laterite hidden under sand which affords sparse grazing in rainy season
- Near river mouths soils are salty and favorable for grazing

### Crop Calendars

**FAO/GIEWS Crop Calendar** <sup>[4]</sup>

<img src="https://www.fao.org/giews/countrybrief/country/SEN/graphics/1_2022-11-07.jpg" />


**USDA/FAS Crop Calendar** <sup>[5]</sup>

<img src="https://ipad.fas.usda.gov/countrysummary/images/SG/cropcalendar/wafrica_sg_calendar.png" />

**Crop Calendar Summary Comparison**
- Both crop calendars include: groundnut, maize, millet, sorghum, and rice.
- The USDA FAS crop calendar also includes cotton.
- Both crop calendars show a sowing/planting period from June to August
- Both crop calendars show a harvest period from September to February

### FAO Stat Ag Land Use (2021) <sup>[6]</sup>
```
Total: 19.671 Mha
└── Agricultural: 9.511 Mha (48.4% of total)
    ├── Cropland: 3.911 Mha (19.9% of total)
    │   ├── Arable: 3.830 Mha (19.5% of total)
    │   │   ├── Temporary crops: 3.3 Mha
    │   │   └── Temporary fallow: 0.53 Mha
    │   └── Permanent crops: 0.081 Mha
    └── Permanent meadows & pastures: 5.6 Mha
```

## 2022 Visual Assessment of Existing Cropland Maps

- [GEE Script Generation](01_2022_visual_assessment_existing.ipynb)
- [GEE Script](https://code.earthengine.google.com/79fb3f16239446a1fbb449bbe8e7c69f)
- [Assessment](https://docs.google.com/spreadsheets/d/1ZlALIwKMC3HzI8-3L4OTEYgBdNjaHKia_0oyAM6qssw/edit?usp=sharing)

## 2022 Intercomparison of Existing Cropland Maps
- [Intercomparison Report](02_2022_intercomparison.ipynb)
- [GEE Script with evaluated Maps for Senegal](https://code.earthengine.google.com/88df189c52f0d19ef07c7fcbca997e51)
- **Best performing map for 2022**: Worldcover v200

## 2022 Crop Map Generation
- [Random Forest Script](https://code.earthengine.google.com/85e02a1e5fbe25d3381b3b7b011395e8)
- [View generated maps](https://code.earthengine.google.com/b0ba4c68291c70229bccffd5de011ba1)
