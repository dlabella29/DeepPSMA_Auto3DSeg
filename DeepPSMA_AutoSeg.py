from monai.apps.auto3dseg import AutoRunner

def main():
   input_config = {
       "modality": "ct",
       "dataroot": "",
       "datalist": "/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/FDG_5fold_train.json",
       "sigmoid": False,
       "resample": False,
       "auto_scale_roi": False,
       "include_background": False,
       "num_epochs": 400,
       "roi_size": [160, 128, 272],
       "amp": True,
       "auto_scale_allowed": False,
       "use_amp": True,
       "class_names": ["ttb","phys"],
 }
   runner = AutoRunner(input=input_config, algos = "segresnet", work_dir= "./FDG_workdir_phys_PETonly")
   runner.run()

if __name__ == '__main__':
  main()

#tensorboard --bind_all --logdir=/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/DeepPSMAAutoSeg/PSMA_workdir_phys_PETonly
#watch -n 1 nvidia-smi


"""
sudo docker run -it --rm --gpus all --shm-size=32g --ipc=host \
  -v "/media/dlabella29/Extreme Pro/Auto_AIMS_Host/input/images/t1-brain-mri:/input/images/t1-brain-mri:ro" \
  -v "/media/dlabella29/Extreme Pro/Auto_AIMS_Host/output/images/tbi-segmentation:/output/images/tbi-segmentation:rw" \
  --user root --entrypoint /bin/bash aimstbi1:101
"""

"""
sudo docker save aimstbi1 | gzip > aimstbi1_DL.tar.gz
"""

"""
watch -n 2 ls -lh "aimstbi.tar.gz"
"""