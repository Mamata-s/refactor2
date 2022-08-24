import os



# # Downsample Original factor 2 (Done)

# model_name= 'dense'
# factor = 2
# checkpoint='outputs/resolution_dataset25/srdense/canny_z_axis25_mask_training_addition_f2_105_0.0001/checkpoints/z_axis/factor_2/epoch_100_f_2.pth'
# # checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_original_f2_165_0.00005/checkpoints/z_axis/factor_2/epoch_160_f_2.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='canny'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# label_edges_dir = f'test_set/label_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir} --label-edges-dir={label_edges_dir}")


# model_name ='dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 2
# checkpoint='outputs/resolution_dataset25/srdense/canny_z_axis25_mask_training_addition_f2_105_0.0001/checkpoints/z_axis/factor_2/epoch_100_f_2.pth'
# # checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_original_f2_165_0.00005/checkpoints/z_axis/factor_2/epoch_160_f_2.pth'
# edge_type='canny'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")



# # Downsample Original factor 4 (Done)

# model_name= 'dense'
# factor = 4
# # checkpoint = 'outputs/resolution_dataset25/srdense/hrdownsample_z_axis25_training_original_f4_805_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_original_f4_165_0.00005/checkpoints/z_axis/factor_4/epoch_160_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# label_edges_dir = f'test_set/label_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir} --label-edges-dir={label_edges_dir}")


# model_name ='dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 4
# checkpoint = 'outputs/resolution_dataset25/srdense/hrdownsample_z_axis25_training_original_f4_805_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# # checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_original_f4_165_0.00005/checkpoints/z_axis/factor_4/epoch_160_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")






# # Downsample Addition factor 2 (Done) 

# model_name= 'dense'
# factor = 2
# checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_addition_f2_165_0.00005/checkpoints/z_axis/factor_2/epoch_160_f_2.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# label_edges_dir = f'test_set/label_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir} --label-edges-dir={label_edges_dir}")


# model_name ='dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 2
# checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_addition_f2_165_0.00005/checkpoints/z_axis/factor_2/epoch_160_f_2.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# # Downsample Addition factor 4 (Done) 

# model_name= 'dense'
# factor = 2
# checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_addition_f4_165_0.00005/checkpoints/z_axis/factor_4/epoch_160_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# label_edges_dir = f'test_set/label_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir} --label-edges-dir={label_edges_dir}")


# model_name ='dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_addition_f4_165_0.00005/checkpoints/z_axis/factor_4/epoch_160_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")





# # Downsample Original factor 4 (Done)

# model_name= 'dense'
# factor = 4
# checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_original_f4_165_0.00005/checkpoints/z_axis/factor_4/epoch_160_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# model_name ='dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25/srdense/hrdownsample_largenet_mseloss_z_axis25_mask_training_original_f4_165_0.00005/checkpoints/z_axis/factor_4/epoch_160_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")








#**********************************************************************************************************************************************************************************


# Model is loaded from densenet_new (BEGIN)


# No mask Original

# model_name= 'dense'
# factor = 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_no_mask_training_original_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# model_name ='dense'
# label_path ='resolution_dataset25_small4/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_no_mask_training_original_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# No mask Addition

# model_name= 'dense'
# factor = 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_no_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# model_name ='dense'
# label_path ='resolution_dataset25_small4/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_no_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# Mask Original
# model_name= 'dense'
# factor = 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_mask_training_original_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# model_name ='dense'
# label_path ='resolution_dataset25_small4/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_mask_training_original_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# Mask Addition
# model_name= 'dense'
# factor = 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# model_name ='dense'
# label_path ='resolution_dataset25_small4/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/hrdownsample_z_axis25_small4_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# Model is loaded from densenet_new (END)




# **********************************************************************************************************************************************************************************************
# **********************************************************************************************************************************************************************************************


# Model is loaded from densenet_smchannel (BEGIN) trained by standardizing the label edges


# NO MASK ORIGINAL
# model_name= 'dense'
# factor = 4
# checkpoint = 'outputs/resolution_dataset25_small4/srdense/edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_no_mask_training_original_f4_505_0.0001/checkpoints/z_axis/factor_4/epoch_200_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

# model_name ='dense'
# label_path ='resolution_dataset25_small4/z_axis/label/test'
# factor= 4
# checkpoint = 'outputs/resolution_dataset25_small4/srdense/edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_no_mask_training_original_f4_505_0.0001/checkpoints/z_axis/factor_4/epoch_200_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# NO MASK ADDITION
# model_name= 'dense'
# factor = 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_no_mask_training_addition_f4_505_0.0001/checkpoints/z_axis/factor_4/epoch_150_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

# model_name ='dense'
# label_path ='resolution_dataset25_small4/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_no_mask_training_addition_f4_505_0.0001/checkpoints/z_axis/factor_4/epoch_150_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# MASK ORIGINAL
# model_name= 'dense'
# factor = 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_mask_training_original_f4_155_0.0001/checkpoints/z_axis/factor_4/epoch_140_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

# model_name ='dense'
# label_path ='resolution_dataset25_small4/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_mask_training_original_f4_155_0.0001/checkpoints/z_axis/factor_4/epoch_140_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# #MASK ADDITION

# model_name= 'dense'
# factor = 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_mask_training_addition_f4_155_0.0001/checkpoints/z_axis/factor_4/epoch_140_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

# model_name ='dense'
# label_path ='resolution_dataset25_small4/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25_small4/srdense/edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_mask_training_addition_f4_155_0.0001/checkpoints/z_axis/factor_4/epoch_140_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")




# Model is loaded from densenet_smchannel (END)


# **********************************************************************************************************************************************************************************************
# **********************************************************************************************************************************************************************************************


# Model is loaded from densenet_smchannel (BEGIN) Trained with Gaussian Edges


# NO MASK ORIGINAL
model_name= 'dense'
factor = 4
checkpoint='outputs/gaussian_dataset25/srdense/gaussian_edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_no_mask_training_original_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
plot_dir= f'test_set/plots_{factor}/'
preds_dir=f'test_set/preds_{factor}/'
edge_type='downsample'
pred_edges_dir=f'test_set/preds_edges_{factor}/'
input_edges_dir=f'test_set/input_edges_{factor}/'
os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

# model_name ='dense'
# label_path ='gaussian_dataset25/z_axis/label/test'
# factor= 4
# checkpoint='outputs/gaussian_dataset25/srdense/gaussian_edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_no_mask_training_original_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# NO MASK ADDITION
# model_name= 'dense'
# factor = 4
# checkpoint='outputs/gaussian_dataset25/srdense/gaussian_edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_no_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

# model_name ='dense'
# label_path ='gaussian_dataset25/z_axis/label/test'
# factor= 4
# checkpoint='outputs/gaussian_dataset25/srdense/gaussian_edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_no_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# MASK ORIGINAL
# model_name= 'dense'
# factor = 4
# checkpoint='outputs/gaussian_dataset25/srdense/gaussian_edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_mask_training_original_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

# model_name ='dense'
# label_path ='gaussian_dataset25/z_axis/label/test'
# factor= 4
# checkpoint='outputs/gaussian_dataset25/srdense/gaussian_edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_mask_training_original_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# #MASK ADDITION

# model_name= 'dense'
# factor = 4
# checkpoint='outputs/gaussian_dataset25/srdense/gaussian_edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

# model_name ='dense'
# label_path ='gaussian_dataset25/z_axis/label/test'
# factor= 4
# checkpoint='outputs/gaussian_dataset25/srdense/gaussian_edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_mask_training_addition_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")



# Model is loaded from densenet_smchannel (END) Trained with Gaussian Edges



# **********************************************************************************************************************************************************************************************
# **********************************************************************************************************************************************************************************************




