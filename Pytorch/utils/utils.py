import os
import numpy as np

def create_folder(dest_dir:str):
    """
      This function creates a folder. If it exists already I doesn't do anything.

      Parameters:
        - dest_dir: string that has the path where the folder(s) are going to be created
      
      Returns:
        - None
    """
    if not (os.path.exists(dest_dir)):
        try:
            print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if (e.errno != e.errno.EEXIST):
                raise

def write_dict_to_txt(config, name):
  with open(name, "w") as dicti_file:
    for k, v in config.items():
        dicti_file.write(f"{k} : {v}\n")
  print(f"File {name} written!")

def write_list_to_txt(list_values, name):
  with open(name, "w") as dicti_file:
    for value in list_values:
        dicti_file.write(f"{value}\n")
  print(f"File {name} written!")

def read_files_from_directory(files_path, limit=None):
    files = []
    i = 0
    for file_path in files_path:
        files.append(np.load(file_path))
        if (i == limit):
          break
        i += 1
    return np.array(files)