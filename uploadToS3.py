import boto3
from botocore.exceptions import NoCredentialsError

# Initialize the S3 client
s3 = boto3.client('s3')

def upload_to_s3(local_file_path, bucket_name, s3_file_name):
    """
    Uploads a file to an S3 bucket.
    
    Parameters:
        local_file_path (str): Path to the file on your local system.
        bucket_name (str): Name of your S3 bucket.
        s3_file_name (str): Name to save the file as in S3.
    """
    try:
        s3.upload_file(local_file_path, bucket_name, s3_file_name)
        print(f"File {s3_file_name} uploaded to bucket {bucket_name}.")
    except FileNotFoundError:
        print("The file was not found.")
    except NoCredentialsError:
        print("Credentials not available.")

# Example usage
local_file_path = 'path/to/your/local/file.csv'
bucket_name = 'your-bucket-name'
s3_file_name = 'data/file.csv'

upload_to_s3(local_file_path, bucket_name, s3_file_name)
