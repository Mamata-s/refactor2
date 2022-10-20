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
# model_name= 'dense'
# factor = 4
# checkpoint='outputs/gaussian_dataset25/srdense/gaussian_edge_range_corr_dense_bottlenecksmall_hrdownsample_z_axis25_small4_no_mask_training_original_f4_105_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

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

# Gaussian filtered lr images and training done on images

# model_name= 'dense'
# factor = 2
# image_path = 'gaussian_dataset25_mul/z_axis/factor_2/test'
# dictionary_path = 'gaussian_dataset25_mul/test_annotation.pkl'
# checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_mul_f2_105/checkpoints/z_axis/factor_2/epoch_250_f_2.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --dictionary-path={dictionary_path} --image-path={image_path} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} ")


# model_name ='dense'
# factor = 2
# label_path ='gaussian_dataset25_mul/z_axis/label/test'
# image_path = 'gaussian_dataset25_mul/z_axis/factor_2/test'
# annotation_path = 'gaussian_dataset25_mul/test_annotation.pkl'
# checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_mul_f2_105/checkpoints/z_axis/factor_2/epoch_250_f_2.pth'
# os.system(f"python eval1.py --checkpoint={checkpoint}  --factor={factor} --model-name={model_name} --image-path={image_path} --label-path={label_path} --annotation-path={annotation_path} ")


# model_name= 'dense'
# factor = 4
# image_path = 'gaussian_dataset25_mul/z_axis/factor_4/test'
# dictionary_path = 'gaussian_dataset25_mul/test_annotation.pkl'
# checkpoint = 'outputs/gaussian_dataset25_mul/patch_gan/patch_gan_dense_wo_perceptual_loss_4/checkpoints/z_axis/factor_4/epoch_300_f_4.pth'
# # checkpoint='outputs/gaussian_dataset25_mul/srdense/gaussian_mul_f24105/checkpoints/z_axis/factor_4/epoch_250_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/patch_gan/patch_gan_dense_wo_perceptual_loss/checkpoints/z_axis/factor_4/epoch_300_f_4.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --dictionary-path={dictionary_path} --image-path={image_path} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} ")


# model_name ='dense'
# factor = 4
# label_path ='gaussian_dataset25_mul/z_axis/label/test'
# image_path = 'gaussian_dataset25_mul/z_axis/factor_4/test'
# downsample_path = 'gaussian_dataset25_mul/z_axis/factor_6/test'
# annotation_path = 'gaussian_dataset25_mul/test_annotation.pkl'
# # checkpoint = 'outputs/gaussian_dataset25_mul/patch_gan/patch_gan_dense_wo_perceptual_loss_4/checkpoints/z_axis/factor_4/epoch_300_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_mul_f2_105/checkpoints/z_axis/factor_2/epoch_250_f_2.pth'
# checkpoint='outputs/gaussian_dataset25_mul/srdense/gaussian_mul_f24105/checkpoints/z_axis/factor_4/epoch_250_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/patch_gan/patch_gan_dense_wo_perceptual_loss/checkpoints/z_axis/factor_4/epoch_300_f_4.pth'
# os.system(f"python eval1.py --checkpoint={checkpoint}  --factor={factor} --model-name={model_name} --image-path={image_path} --label-path={label_path} --annotation-path={annotation_path} --downsample-path={downsample_path}")





# TRAINED ON GAUSSIAN EDGES



# model_name= 'dense'
# factor = 2
# checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_no_mask_original_2/checkpoints/z_axis/factor_2/epoch_100_f_2.pth'

# # checkpoint='outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_no_mask_original/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_no_mask_addition/checkpoints/z_axis/factor_4/epoch_150_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_mask_addition_wo_normalization/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/patch_gan/patch_gan_dense_edges/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/patch_gan/patch_gan_dense_edges_addition/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'

# # checkpoint = 'outputs/gaussian_dataset25_mul/patch_gan/patch_gan_dense_perceptual_loss/checkpoints/z_axis/factor_4/epoch_2_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_no_mask_addition_2/checkpoints/z_axis/factor_2/epoch_100_f_2.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# dictionary_path = 'gaussian_dataset25_mul/test_annotation.pkl'
# image_path = 'gaussian_dataset25_mul/z_axis/factor_2/test'
# downsample_path = 'gaussian_dataset25_mul/z_axis/factor_4/test'
# os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir} --image-path={image_path} --dictionary-path={dictionary_path} --downsample-path={downsample_path}")


# model_name ='dense'
# label_path ='gaussian_dataset25_mul/z_axis/label/test'
# image_path = 'gaussian_dataset25_mul/z_axis/factor_4/test'
# factor= 4
# annotation_path = 'gaussian_dataset25_mul/test_annotation.pkl'

# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_no_mask_original_2/checkpoints/z_axis/factor_2/epoch_100_f_2.pth'

# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_mask_addition_wo_normalization/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_mask_addition/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'

# checkpoint='outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_no_mask_original/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_no_mask_addition/checkpoints/z_axis/factor_4/epoch_150_f_4.pth'


# # checkpoint = 'outputs/gaussian_dataset25_mul/patch_gan/patch_gan_dense_edges/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'

# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_mask_addition_wo_normalization/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'

# # checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_dataset25_mul_no_mask_addition_2/checkpoints/z_axis/factor_2/epoch_100_f_2.pth'
# edge_type='downsample'
# downsample_path = 'gaussian_dataset25_mul/z_axis/factor_6/test'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --annotation-path={annotation_path} --factor={factor} --label-path={label_path} --image-path={image_path} --edges --edge-type={edge_type}  --downsample-path={downsample_path}")




# GAUSSIAN IMAGE TRAINING DENSENET FOR SIGMA 50,75,100
# # sigma100
# checkpoint= 'outputs/gaussian_dataset25_sigma100/srdense/gaussian_mul_wo_circular_mask_sigma100/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

# model_name= 'dense'
# factor = 2
# image_path = 'upsample/upsample_image_sigma_150'
# dictionary_path = 'upsample/annotation_hr1_dict.pkl'

# image_path = 'gaussian_dataset25_sigma50/z_axis/factor_2/test'
# dictionary_path = 'gaussian_dataset25_sigma50/test_annotation.pkl'

# checkpoint = 'outputs/gaussian_dataset25_mul/srdense/gaussian_mul_f2_105/checkpoints/z_axis/factor_2/epoch_250_f_2.pth'
# checkpoint = 'outputs/gaussian_dataset25_mul_wo_circular_mask/srdense/gaussian_mul_wo_circular_mask/checkpoints/z_axis/factor_2/epoch_1000_f_2.pth'

# checkpoint= 'outputs/gaussian_dataset25_sigma150/srdense/gaussian_mul_wo_circular_mask_sigma150/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# checkpoint ='outputs/gaussian_dataset25_sigma75/srdense/gaussian_mul_wo_circular_mask_sigma75/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

# checkpoint ='outputs/gaussian_dataset25_sigma75/srdense/gaussian_mul_wo_circular_mask_sigma75/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

# checkpoint = 'outputs/gaussian_dataset25_sigma125/srdense/gaussian_mul_wo_circular_mask_sigma125/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

# # checkpoint ='outputs/gaussian_dataset25_mul/srdense/combined_model/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'

# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --dictionary-path={dictionary_path} --image-path={image_path} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} ")



# model_name ='dense'
# label_path ='upsample/hr_25/z_axis'
# image_path = 'upsample/upsample_image_sigma_150'
# factor= 2
# annotation_path = 'upsample/annotation_hr2_dict.pkl'
# # checkpoint ='outputs/gaussian_dataset25_sigma50/srdense/gaussian_mul_wo_circular_mask_sigma50/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# checkpoint = 'outputs/gaussian_dataset25_sigma125/srdense/gaussian_mul_wo_circular_mask_sigma125/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --annotation-path={annotation_path} --factor={factor} --label-path={label_path} --image-path={image_path}")




# Checking for combined model (model predicting both image and edges)
# model_name= 'dense'
# factor = 2
# image_path = 'upsample/f1_160/upsample_image'
# dictionary_path = 'upsample/f1_160/annotation_hr1_dict.pkl'
# checkpoint = 'outputs/gaussian_dataset25_mul/srdense/combined_model_high_freq_edges/checkpoints/z_axis/factor_4/epoch_120_f_4.pth'
# edge_type = 'canny'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --dictionary-path={dictionary_path} --image-path={image_path} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type}  ")



# GAUSSIAN IMAGE TRAINING DENSENET FOR SIGMA 50,75,100

model_name= 'dense'
factor = 2
# dictionary_path = 'upsample/combine/annotation_hr1_dict.pkl'
dictionary_path = 'dataset_images/gaussian_dataset25_sigma100/annotation.pkl'
plot_dir= f'test_set/plots_{factor}/'
preds_dir=f'test_set/preds_{factor}/'

# image_path = 'upsample/combine/upsample_sigma_50'
# checkpoint='outputs/gaussian_dataset25_sigma50/srdense/gaussian_mul_wo_circular_mask_sigma50/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

# image_path = 'upsample/combine/upsample_sigma_75'
# checkpoint='outputs/gaussian_dataset25_sigma75/srdense/gaussian_mul_wo_circular_mask_sigma75/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

# image_path = 'upsample/combine/upsample_sigma_100'
image_path = 'dataset_images/gaussian_dataset25_sigma100/z_axis/factor_2/train'
checkpoint='outputs/gaussian_dataset25_sigma100/srdense/gaussian_mul_wo_circular_mask_sigma100/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

# image_path = 'upsample/combine/upsample_sigma_125'
# checkpoint='outputs/gaussian_dataset25_sigma125/srdense/gaussian_mul_wo_circular_mask_sigma125/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

# image_path = 'upsample/combine/upsample_sigma_150'
# checkpoint='outputs/gaussian_dataset25_sigma150/srdense/gaussian_mul_wo_circular_mask_sigma150/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

os.system(f"python test2.py --checkpoint={checkpoint} --dictionary-path={dictionary_path} --image-path={image_path} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} ")



# Evaluating Bicubic and gaussian model on 50 micron

# model_name ='dense'
# label_path ='upsample/combine/hr_25/z_axis'
# image_path = 'upsample/combine/upsample_bicubic'
# factor= 2
# annotation_path = 'upsample/combine/annotation_hr2_dict.pkl'
# # checkpoint ='outputs/dataset_images/bicubic_dataset25/srdense/bicubic_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# checkpoint = 'outputs/dataset_images/resolution_dataset25_full/srdense/kspace_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --annotation-path={annotation_path} --factor={factor} --label-path={label_path} --image-path={image_path}")


#testing bicubic and gaussian model on 50 micron

# model_name= 'dense'
# factor = 2
# # image_path = 'upsample/combine/upsample_bicubic'
# image_path = 'upsample/combine/upsample_image'
# dictionary_path = 'upsample/combine/annotation_hr1_dict.pkl'
# # checkpoint ='outputs/dataset_images/bicubic_dataset25/srdense/bicubic_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# checkpoint = 'outputs/dataset_images/resolution_dataset25_full/srdense/kspace_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# plot_dir= f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# os.system(f"python test1.py --checkpoint={checkpoint} --dictionary-path={dictionary_path} --image-path={image_path} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir}")


# # Evaluating bicubic and gaussian on their own test set from 25 micron
# model_name ='dense'
# label_path ='dataset_images/bicubic_dataset25/z_axis/label/test'
# image_path = 'dataset_images/bicubic_dataset25/z_axis/factor_2/test'
# factor= 2
# annotation_path = 'dataset_images/bicubic_dataset25/z_axis/factor_2/annotation_test_dict.pkl'
# checkpoint ='outputs/dataset_images/bicubic_dataset25/srdense/bicubic_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# # checkpoint = 'outputs/dataset_images/resolution_dataset25_full/srdense/kspace_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# os.system(f"python eval1.py --checkpoint={checkpoint} --model-name={model_name} --annotation-path={annotation_path} --factor={factor} --label-path={label_path} --image-path={image_path}")
