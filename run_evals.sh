cd /home/wpai/DP-DINO-Policy
source /home/wpai/miniforge3/bin/activate dp-dino

# Command 1
echo "=== Running evaluation ==="

# python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/dp_mug3_20/dp-mug3-20_ep100.ckpt --output_dir data/eval_dp_muglift3_100_ep100  --env_name MugLift_3  --n_train 20  --dataset_path /home/wpai/robosuite/demo/mug3/mug3_20_image.hdf5


python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/dino-dp-curri_mug5_100/dp-dino-ct-100eps-100epoch.ckpt --output_dir data/eval_dino-dp-curri_muglift5_100_ep100  --env_name MugLift_5  --n_train 20  --dataset_path /home/wpai/robosuite/demo/mug5/mug5_20_image.hdf5
echo "=== All evaluations complete! ==="