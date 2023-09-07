import os

def rename_files_in_directory(root_directory):
    # Loop through each directory, subdirectory, and file in the root directory
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Loop through each file in the current directory
        for filename in filenames:
            # Check if the file has a .c or .h extension
            if filename.endswith(".c") or filename.endswith(".h"):
                # Construct the full path of the file
                full_path = os.path.join(dirpath, filename)
                # Create a new filename with a .txt extension
                new_filename = os.path.splitext(filename)[0] + ".txt"
                new_full_path = os.path.join(dirpath, new_filename)
                # Rename the file
                os.rename(full_path, new_full_path)
                print(f"Renamed: {full_path} -> {new_full_path}")

# The root directory from which to start the search
root_directory = "/content/msdk"

# Call the function to start renaming files
rename_files_in_directory(root_directory)
