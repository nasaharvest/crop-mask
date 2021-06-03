export BUCKET=crop-mask-ee-data

cp -R src/ETL gcp/export_unlabeled_function/src

gcloud functions deploy export-unlabeled \
    --source=gcp/export_unlabeled_function \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python37 \
    --entry-point=export_unlabeled \
    --timeout=300s \
    --set-env-vars DEST_BUCKET=$BUCKET

gcloud functions deploy ee-status \
    --source=gcp/export_unlabeled_function \
    --trigger-http \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=get_status

rm -rf gcp/export_unlabeled_function/src/*