import os
import subprocess

def download_dataset():
    """
    Downloads the malaria dataset from Kaggle.
    Requires Kaggle API credentials to be set up in advance.
    """
    try:
        # Ensure Kaggle API credentials are available
        kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
        if not os.path.exists(kaggle_path):
            raise FileNotFoundError("Kaggle API credentials not found. Please upload 'kaggle.json'.")

        # Set file permissions (required by Kaggle API)
        os.chmod(kaggle_path, 0o600)

        # Download the dataset
        print("Downloading the malaria dataset from Kaggle...")
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'iarunava/cell-images-for-detecting-malaria', '--force'], check=True)

        # Unzip the downloaded file
        print("Unzipping the dataset...")
        subprocess.run(['unzip', '-q', 'cell-images-for-detecting-malaria.zip'], check=True)

        print("Dataset downloaded and extracted successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_dataset()
