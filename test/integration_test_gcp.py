from unittest import TestCase

import json
import requests


class TestGcp(TestCase):
    """
    This is not a real test, these functions simply trigger behavior which can
    then be manually checked on Google Cloud or inside a docker container.
    """

    def test_gcp_export(self):
        url = "http://us-central1-bsos-geog-harvest1.cloudfunctions.net/export-region"
        payload = {
            "model_name": "Ethiopia_Tigray_2020",
            "version": "test1",
            "min_lon": 36.45,
            "max_lon": 36.47,
            "min_lat": 12.25,
            "max_lat": 12.27,
            "start_date": "2020-02-01",
            "end_date": "2021-02-01",
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(response.content)

    def test_inference_in_docker(self):
        url = "http://localhost:8080/predictions/Ethiopia_Tigray_2020"
        response = requests.post(url, data={"uri": "mock-path"})
        print(response.content)
