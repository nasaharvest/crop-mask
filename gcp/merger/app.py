from flask import Flask, request
from google.cloud import storage
import os
from pathlib import Path
import tempfile
import shutil
from osgeo import gdal, osr
from gcp.merger.main_andrew_docker import gdal_build_vrt as build
from gcp.merger.main_andrew_docker import gdal_translate_tif as translate

# from main_andrew import gdal_build_vrt
# from import_nc_files import get_all_blob
app = Flask(__name__)


# download files from given bucket
def download_files(bucket_name, storage_client):
    # create temp_directory
    temp_dir = Path(tempfile.mkdtemp())
    # make sure it exists
    print(temp_dir.exists())
    try:
        counter = 0
        curr_direc = None
        for blob in storage_client.list_blobs(bucket_name):
            print('next ' + str(counter))
            counter += 1
            # set curr_path to blob's local path
            blob_path = Path(blob.name)
            # set curr_direc to temp_file path + parent of curr_path
            curr_direc = temp_dir / blob_path.parent.name
            curr_direc.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(temp_dir / blob_path))
            
        print("Length final directory is " + str(len((os.listdir(str(curr_direc))))))

    except Exception as e:
        print("This was the exception caught: " + str(e))     
    finally:
        print(str(temp_dir))

    print("Files can be found at " + str(temp_dir))
    return temp_dir


# delete given directory
def remove(temp_dir):
    shutil.rmtree(temp_dir)


@app.route('/')
def hello_world():
    return 'Hey, we have Flask in a Docker container!'


@app.route('/download', methods = ['POST'])
def gdalmerge():
    if request.method == 'POST':

        # get info from POST request
        json_data = request.get_json()
        bucket_name =  json_data['bucket_name'] 
        print('Data Received: "{data}"'.format(data=json_data))

        # connect to Google Cloud Storage
        storage_client = storage.Client()

        # download files
        temp_dir = download_files(bucket_name, storage_client)

        # tell where files can be found
        result = "Files can be found at " + str(temp_dir)
        print("The temp directory defintely does not exist: " + str(temp_dir.exists()))
        remove(temp_dir)
        return result
    return "done"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
