from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    @task
    def predict(self):
        uri = (
            "gs://crop-mask-ee-data/Test_Rwanda_full_2020-04-01_2021-04-010000000000-0000000000.tif"
        )
        self.client.post("/predictions/Rwanda", data={"uri": uri})
