from unittest import TestCase

import requests


class TestGcp(TestCase):
    """
    This is not a real test, these functions simply trigger behavior which can
    then be manually checked on Google Cloud or inside a docker container.
    """

    def test_inference_in_docker(self):
        url = "http://localhost:8080/predictions/Ethiopia_Tigray_2020"
        response = requests.post(url, data={"uri": "mock-path"})
        print(response.content)
