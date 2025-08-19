import os
import SimpleITK as sitk

# Define source and target directories
source_folder = '/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/DeepPSMAAutoSeg/input/images/nii'
target_folder = '/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/DeepPSMAAutoSeg/input/images/mha'

# Ensure the target directory exists
os.makedirs(target_folder, exist_ok=True)

# Get a list of all .nii.gz files in the source directory
nii_files = [f for f in os.listdir(source_folder) if f.endswith('.nii.gz')]

# Loop over each .nii.gz file and convert it to .mha
for nii_file in nii_files:
    source_path = os.path.join(source_folder, nii_file)
    # Read the .nii.gz image
    image = sitk.ReadImage(source_path)

    # Prepare the target filename with .mha extension
    base_name = os.path.splitext(os.path.splitext(nii_file)[0])[0]  # Removes .nii.gz extension
    mha_file = base_name + '.mha'
    target_path = os.path.join(target_folder, mha_file)

    # Write the image as .mha
    sitk.WriteImage(image, target_path)
    print(f"Converted {nii_file} to {mha_file}")


