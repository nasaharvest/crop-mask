# Setup for Google Cloud Platform
```bash
gcloud config set project nasa-harvest
gsutil mb gs://crop-mask-ee-data
sh deploy_ee_functions.sh
sh deploy_inference.sh
```
Once GCP is setup you can do the following:
**1. Start inference run:**
```bash
curl -X POST http://us-central1-nasa-harvest.cloudfunctions.net/export-unlabeled \
    -H "Content-Type:application/json" \
    -d @gcp/example.json | python -mjson.tool
```

**2. Seeing EarthEngine progress**
```bash
curl https://us-central1-nasa-harvest.cloudfunctions.net/ee-status?additional=FAILED,COMPLETED | python -mjson.tool
```

**3. Running inference container locally:**
```bash
# Set credentials path to run image locally (this is an example path)
export GCLOUD_CREDENTIALS=$HOME/nasaharvest/secrets/google_application_credentials.json
docker run -v $GCLOUD_CREDENTIALS:/root/.config/google_application_credentials -d -p 8080:8080 -p 8081:8081 $TAG
```

**4. Testing inference container locally:**
```bash
curl http://127.0.0.1:8080/ping     # Basic test
curl http://127.0.0.1:8081/models   # Verify models present
export URI="gs://crop-mask-ee-data/RwandaSake/RwandaSake_2019-04-01_2020-04-01.tif"
curl -X POST -d "uri=$URI" http://127.0.0.1:8080/predictions/<model name>     # Make prediction
```

curl -X POST -d "uri=$URI" http://127.0.0.1:8080/predictions/Rwanda