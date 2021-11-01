from flask import Flask, request
from google.cloud import storage
import os
from pathlib import Path
import tempfile
import shutil
# from main_andrew import gdal_build_vrt
# from import_nc_files import get_all_blob
app = Flask(__name__)



@app.route('/')
def hello_world():
    return 'Hey, we have Flask in a Docker container!'

@app.route('/download', methods = ['POST'])
def gdalmerge():
    if request.method == 'POST':
        print("in the post")
        # create temp_directory
        temp_dir = Path(tempfile.mkdtemp())
        # make sure it exists
        print(temp_dir.exists())
        curr_direc = []
        try:
            # connect to bucket
            storage_client = storage.Client()
            # get data from json
            json_data = request.get_json()
            bucket_name =  json_data['bucket_name'] 
            # print data to verify it was downloaded
            print('Data Received: "{data}"'.format(data=json_data))
            # for each blob
            counter = 0
            for blob in storage_client.list_blobs(bucket_name):
                print('next ' + str(counter))
                counter += 1
                # set curr_path to blob's local path
                curr_path = Path(blob.name)
                # set curr_direc to temp_file path + parent of curr_path
                prev = curr_direc
                if prev != Path(str(temp_dir) + curr_path.parent.name) and prev != []:
                    print("The length of the current directory is " + str(len(os.listdir(str(curr_direc)))))
                curr_direc = Path(str(temp_dir) + curr_path.parent.name)
                
                # if curr_direct exists then download file there 
                if curr_direc.exists():
                    blob.download_to_filename(str(temp_dir) + str(curr_path))
                # if it doesn't create directory there, then download
                else:
                    os.makedirs(curr_direc, exist_ok=True)   
                    blob.download_to_filename(str(temp_dir) + str(curr_path)) 
                  
        # remove files  
        finally:
            print("The length of the final directory is " + str(len((os.listdir(str(curr_direc))))))
            shutil.rmtree(temp_dir)

        return "yessir"
    return "done"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

