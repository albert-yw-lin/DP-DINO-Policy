cd /home/wpai/DP-DINO-Policy
source /home/wpai/miniforge3/bin/activate dp-dino

# Command 1
echo "=== Running evaluation 1/3 ==="
python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/2025.12.03/23.49.07_train_diffusion_unet_image_mug_lift_image/checkpoints/latest.ckpt --output_dir data/eval_dp_muglift3_seed  --env_name MugLift_3  --n_test 20  --dataset_path /home/wpai/robosuite/demo/mug3/mug3_20_image.hdf5

# Command 2
echo "=== Running evaluation 2/3 ==="
python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/2025.12.03/23.49.07_train_diffusion_unet_image_mug_lift_image/checkpoints/latest.ckpt --output_dir data/eval_dp_muglift4_seed  --env_name MugLift_4  --n_test 20  --dataset_path /home/wpai/robosuite/demo/mug4/mug4_20_image.hdf5

# Command 3
echo "=== Running evaluation 3/3 ==="
python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/2025.12.03/23.49.07_train_diffusion_unet_image_mug_lift_image/checkpoints/latest.ckpt --output_dir data/eval_dp_muglift5_seed  --env_name MugLift_5  --n_test 20  --dataset_path /home/wpai/robosuite/demo/mug5/mug5_20_image.hdf5

echo "=== Running evaluation 4/3 ==="
python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/2025.12.03/23.49.07_train_diffusion_unet_image_mug_lift_image/checkpoints/latest.ckpt --output_dir data/eval_dp_muglift2_seed  --env_name MugLift_2  --n_test 20  --dataset_path /home/wpai/robosuite/demo/mug2/mug2_20_image.hdf5
# ... continue for all 9 commands
echo "=== Running dino-dp evaluation 1/4 ==="
python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/2025.12.03/dino-dp_mug80/dp-dino-mug-80eps.ckpt --output_dir data/eval_dino-dp_muglift2_seed  --env_name MugLift_2  --n_test 20  --dataset_path /home/wpai/robosuite/demo/mug2/mug2_20_image.hdf5
echo "=== Running dino-dp evaluation 2/4 ==="
python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/2025.12.03/dino-dp_mug80/dp-dino-mug-80eps.ckpt --output_dir data/eval_dino-dp_muglift3_seed  --env_name MugLift_3  --n_test 20  --dataset_path /home/wpai/robosuite/demo/mug3/mug3_20_image.hdf5
echo "=== Running dino-dp evaluation 3/4 ==="
python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/2025.12.03/dino-dp_mug80/dp-dino-mug-80eps.ckpt --output_dir data/eval_dino-dp_muglift4_seed  --env_name MugLift_4  --n_test 20  --dataset_path /home/wpai/robosuite/demo/mug4/mug4_20_image.hdf5
echo "=== Running dino-dp evaluation 4/4 ==="
python eval.py --checkpoint /home/wpai/DP-DINO-Policy/data/outputs/2025.12.03/dino-dp_mug80/dp-dino-mug-80eps.ckpt --output_dir data/eval_dino-dp_muglift5_seed  --env_name MugLift_5  --n_test 20  --dataset_path /home/wpai/robosuite/demo/mug5/mug5_20_image.hdf5


echo "=== All evaluations complete! ==="