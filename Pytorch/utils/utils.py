import os

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