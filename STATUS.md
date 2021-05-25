# Crop Map Generation Status

**Steps for generating crop map:**
1. Data labeled on CEO (for training)
2. Initial model trained ([see metrics](data/model_metrics.json))
3. Data for inference exported from Google Earth Engine
4. Initial crop map generated
5. Expert analysis + corrective labeling complete on CEO
6. Model retrained with corrective labels
7. Corrected crop map generated (can go back to step 5 here)
8. Generated test dataset
9. Recorded test set metrics
10. Crop map uploaded to Zenodo

[Step 0]: https://progress-bar.dev/0/?scale=10&suffix=/10&width=400
[Step 1]: https://progress-bar.dev/1/?scale=10&suffix=/10&width=400
[Step 2]: https://progress-bar.dev/2/?scale=10&suffix=/10&width=400
[Step 3]: https://progress-bar.dev/3/?scale=10&suffix=/10&width=400
[Step 4]: https://progress-bar.dev/4/?scale=10&suffix=/10&width=400
[Step 5]: https://progress-bar.dev/5/?scale=10&suffix=/10&width=400
[Step 6]: https://progress-bar.dev/6/?scale=10&suffix=/10&width=400
[Step 7]: https://progress-bar.dev/7/?scale=10&suffix=/10&width=400
[Step 8]: https://progress-bar.dev/8/?scale=10&suffix=/10&width=400
[Step 9]: https://progress-bar.dev/9/?scale=10&suffix=/10&width=400
[Step 10]: https://progress-bar.dev/10/?scale=10&suffix=/10&width=400

[Kenya Zenodo]: https://zenodo.org/record/4271144#.YK07oJNKhTZ
[Togo Zenodo]: https://zenodo.org/record/3836629#.YK08FJNKhTY

|Country            |Season         |Steps Complete |Link   |
|---                |:---:          |:---:          |:---:  |
|Kenya              |2019/20        |![Step 10]     |[Kenya Zenodo]   |
|Togo               |2019/20        |![Step 10]     |[Togo Zenodo]   |
|Mali (lower USAID) |2020/21        |![Step 4]      |       |
|Mali (upper USAID) |2020/21        |![Step 4]      |       |
|Rwanda             |2019/20        |![Step 4]      |       |
|Uganda             |2020/21        |![Step 0]      |       |
|Uganda (Maize)     |2020/21        |![Step 0]      |       |
