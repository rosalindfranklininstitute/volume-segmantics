import subprocess
import argparse
import os

def model_predict_2d(model_path, folder_path):
    for i in os.listdir(folder_path):
        if not (i.lower().endswith('.tif') or i.lower().endswith('.tiff')):
            continue
        
     
        full_path = os.path.join(folder_path, i)
        if not os.path.isfile(full_path):
            continue
        
        command = ['model-predict-2d', model_path, full_path]
        try:
            
            result = subprocess.run(command, text=True)
        
        except Exception as e:
            return f"An error occurred while running the prediction: {str(e)}"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="prediction on each tif"
    )
    parser.add_argument(
        "-mp", "--model_path", required=True, help="path to model file"
    )
    parser.add_argument(
        "-fp", "--folder_path", required=True, help="Folder containing tifs"
    )
    args = parser.parse_args()
    call_func = model_predict_2d(args.model_path, args.folder_path)

if __name__ == '__main__':
    main()