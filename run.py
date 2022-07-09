import os

# DONE
# # ******* Get result of  downsample edges with losstype original ie loss calculate on edges
# model_name= 'dense'
# factor=2
# checkpoint='outputs/resolution_dataset50/srdense/SRDENSE_GR7NB5NL5S4_DOWNSAMPLE_EDGES_MSE_ADAM_F2_LR0.001_FULL50_EP5005_initNorm/checkpoints/full/factor_2/epoch_5000_f_2.pth'
# plot_dir='test_set/plots_2/'
# preds_dir='test_set/preds_2/'
# edge_type='downsample'
# pred_edges_dir='test_set/preds_edges_2/'
# input_edges_dir='test_set/input_edges_2/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


#***************Get the result of downsample edges patch with loss type addition on edges
# model_name= 'dense'
# factor=2
# checkpoint='outputs/resolution_dataset50/srdense/dense_z_axis_mask_training/checkpoints/z_axis/factor_2/epoch_18_f_2.pth'
# # checkpoint='outputs/resolution_dataset50/srdense/PATCH_SRDENSE_NB5_KAI_NE605_BS32_LR0.0005_P96_F2_ADAM_MSE_NEPOCH5005_ltypeADD/checkpoints/patch/patch-96/factor_2/epoch_200_f_2.pth'
# plot_dir='test_set/plots_2/'
# preds_dir='test_set/preds_2/'
# edge_type='downsample'
# pred_edges_dir='test_set/preds_edges_2/'
# input_edges_dir='test_set/input_edges_2/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")



# DONE
# ******* Get result of  downsample edges with losstype ADDITION ie loss calculate on images
# model_name= 'dense'
# factor=2
# checkpoint='outputs/resolution_dataset50/srdense/SRDENSE_GR7NB5NL5S4_DOWNSAMPLE_EDGES_LtypeAdd_MSE_ADAM_F2_LR0.001_FULL50_EP805_initNorm/checkpoints/full/factor_2/epoch_100_f_2.pth'
# plot_dir='test_set/plots_2/'
# preds_dir='test_set/preds_2/'
# edge_type='downsample'
# pred_edges_dir='test_set/preds_edges_2/'
# input_edges_dir='test_set/input_edges_2/'

# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")



# DONE
# **************** to get result for srdense net trained on images(not edges)***************************
# model_name= 'dense'
# factor=4
# checkpoint='outputs/resolution_dataset50/srdense/SRDENSE_GR7NB5NL5S4_KAI_NE3005_BS32_LR0.006_Z_F4_ADAM_MSE/checkpoints/z_axis/factor_4/epoch_3000_f_4.pth'
# plot_dir='test_set/plots_4/'
# preds_dir='test_set/preds_4/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} ")


# DONE
# **************** to get result for srdense net trained on images(not edges)***************************
# model_name= 'dense'
# factor=2
# checkpoint='outputs/resolution_dataset50/srdense/SRDENSE_GR7NB5NL5S4_KAI_NE3005_BS32_LR0.006_Z_F2_ADAM_MSE/checkpoints/z_axis/factor_2/epoch_1700_f_2.pth'
# plot_dir ='test_set/plots_2/'
# preds_dir ='test_set/preds_2/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} ")


# DONE
#******************** To get the result of srdense_edge_training_4 (this model predicts only edges) and trained on canny edges with loss type original(without addition)
# model_name= 'dense'
# factor=4
# checkpoint='outputs/resolution_dataset50/srdense/srdense_edge_training_4/checkpoints/z_axis/factor_4/epoch_900_f_4.pth'
# plot_dir='test_set/plots_4/'
# preds_dir='test_set/preds_4/'
# edge_type='canny'
# pred_edges_dir='test_set/preds_edges_4/'
# input_edges_dir='test_set/input_edges_4/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# Done
#******************** To get the result of srdense_edge_training_2 (this model predicts only edges) and trained on canny edges with loss type original(without addition)
# model_name= 'dense'
# factor=2
# checkpoint='outputs/resolution_dataset50/srdense/srdense_edge_training_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
# plot_dir='test_set/plots_2/'
# preds_dir='test_set/preds_2/'
# edge_type='canny'
# pred_edges_dir='test_set/preds_edges_2/'
# input_edges_dir='test_set/input_edges_2/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# DONE
# *********************** To get the result of srdense_edge_ltype_addition_2 (trained on canny edges)
# model_name= 'dense'
# factor=2
# checkpoint='outputs/resolution_dataset50/srdense/srdense_edge_ltype_addition_2/checkpoints/z_axis/factor_2/epoch_1500_f_2.pth'
# plot_dir='test_set/plots_2/'
# preds_dir='test_set/preds_2/'
# edge_type='canny'
# pred_edges_dir='test_set/preds_edges_2/'
# input_edges_dir='test_set/input_edges_2/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# Done
# *********************** To get the result of srdense_edge_ltype_addition_4 (trained on canny edges)
# model_name= 'dense'
# factor=4
# checkpoint='outputs/resolution_dataset50/srdense/srdense_edge_ltype_addition_4/checkpoints/z_axis/factor_4/epoch_1500_f_4.pth'
# plot_dir='test_set/plots_4/'
# preds_dir='test_set/preds_4/'
# edge_type='canny'
# pred_edges_dir='test_set/preds_edges_4/'
# input_edges_dir='test_set/input_edges_4/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")





#***************************** EVALUATION **********************************************************************************************************


# # Canny Edges Original factor 2 (DONE)
# model_name= 'dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 2
# checkpoint='outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_original_f2_155_0.0001/checkpoints/z_axis/factor_2/epoch_150_f_2.pth'
# edge_type='canny'
# os.system(f"python eval.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# Canny Edges Original factor 4 (Done)
# model_name= 'dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 4
# checkpoint = 'outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_original_f4_125_0.0001_gaussian_maskingcorr/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# # checkpoint='outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_original_f4_125_0.0001/checkpoints/z_axis/factor_4/epoch_120_f_4.pth'
# edge_type='canny'
# os.system(f"python eval.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# # Canny Edges Addition factor 2  (DONE)
# model_name= 'dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 2
# checkpoint='outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_addition_f2_105_0.0001_sgd/checkpoints/z_axis/factor_2/epoch_100_f_2.pth'
# edge_type='canny'
# os.system(f"python eval.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# # Canny Edges Addition factor 4(Done)
# model_name= 'dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 4
# checkpoint='outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_addition_f4_105_0.0001_sgd/checkpoints/z_axis/factor_4/epoch_80_f_4.pth'
# edge_type='canny'
# os.system(f"python eval.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")




#Downsample Edges Original Factor 2  (done)
# model_name= 'dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 2
# checkpoint='outputs/resolution_dataset25/srdense/dense_downsample_z_axis25_mask_training_original_f2_805_0.0001/checkpoints/z_axis/factor_2/epoch_300_f_2.pth'
# edge_type='downsample'
# os.system(f"python eval.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


# #Downsample Edges Original Factor 4 (DONE)
# model_name= 'dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 4
# checkpoint= 'outputs/resolution_dataset25/srdense/dense_downsample_z_axis25_mask_training_original_f4_805_0.0001/checkpoints/z_axis/factor_4/epoch_300_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


#Downsample Edges Addition Factor 2 (DONE)
# model_name= 'dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 2
# checkpoint='outputs/resolution_dataset25/srdense/dense_downsample_z_axis25_mask_training_addition_f2_105_0.0001/checkpoints/z_axis/factor_2/epoch_80_f_2.pth'
# edge_type='downsample'
# os.system(f"python eval.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")


#Downsample Edges Addition Factor 4 (Done)
# model_name= 'dense'
# label_path ='resolution_dataset25/z_axis/label/test'
# factor= 4
# checkpoint = 'outputs/resolution_dataset25/srdense/dense_downsample_z_axis25_mask_training_addition_f4_805_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# edge_type='downsample'
# os.system(f"python eval.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --label-path={label_path} --edge-type={edge_type} ")



#***************************** TEST **********************************************************************************************************

# # ******* Canny Edges Original factor 2 (Done)
model_name= 'dense'
factor=2
checkpoint='outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_original_f2_155_0.0001/checkpoints/z_axis/factor_2/epoch_150_f_2.pth'
plot_dir='test_set/plots_2/'
preds_dir='test_set/preds_2/'
edge_type='canny'
pred_edges_dir='test_set/preds_edges_2/'
input_edges_dir='test_set/input_edges_2/'
os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# ******* Canny Edges Original factor 4 (Done)
# model_name= 'dense'
# factor=4
# checkpoint = 'outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_original_f4_125_0.0001_gaussian_maskingcorr/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# # checkpoint='outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_original_f4_125_0.0001/checkpoints/z_axis/factor_4/epoch_120_f_4.pth'
# plot_dir='test_set/plots_4/'
# preds_dir='test_set/preds_4/'
# edge_type='canny'
# pred_edges_dir='test_set/preds_edges_4/'
# input_edges_dir='test_set/input_edges_4/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# # ******* Canny Edges Addition factor 2 (Done)
# model_name= 'dense'
# factor=2
# checkpoint='outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_addition_f2_105_0.0001_sgd/checkpoints/z_axis/factor_2/epoch_100_f_2.pth'
# plot_dir='test_set/plots_2/'
# preds_dir='test_set/preds_2/'
# edge_type='canny'
# pred_edges_dir='test_set/preds_edges_2/'
# input_edges_dir='test_set/input_edges_2/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")

# ******* Canny Edges Addition factor 4 (Done)
# model_name= 'dense'
# factor=4
# checkpoint='outputs/resolution_dataset25/srdense/dense_canny_z_axis25_mask_training_addition_f4_105_0.0001_sgd/checkpoints/z_axis/factor_4/epoch_80_f_4.pth'
# plot_dir='test_set/plots_4/'
# preds_dir='test_set/preds_4/'
# edge_type='canny'
# pred_edges_dir='test_set/preds_edges_4/'
# input_edges_dir='test_set/input_edges_4/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")





# # Downsample Original factor 2 (Done)
# model_name= 'dense'
# factor=2
# checkpoint='outputs/resolution_dataset25/srdense/dense_downsample_z_axis25_mask_training_original_f2_805_0.0001/checkpoints/z_axis/factor_2/epoch_300_f_2.pth'
# plot_dir='test_set/plots_2/'
# preds_dir='test_set/preds_2/'
# edge_type='downsample'
# pred_edges_dir='test_set/preds_edges_2/'
# input_edges_dir='test_set/input_edges_2/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")



# Downsample Original factor 4 (Done)
# model_name= 'dense'
# factor=4
# checkpoint= 'outputs/resolution_dataset25/srdense/dense_downsample_z_axis25_mask_training_original_f4_805_0.0001/checkpoints/z_axis/factor_4/epoch_300_f_4.pth'
# plot_dir='test_set/plots_4/'
# preds_dir='test_set/preds_4/'
# edge_type='downsample'
# pred_edges_dir='test_set/preds_edges_4/'
# input_edges_dir='test_set/input_edges_4/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# # Downsample Addition factor 2 (Done)
# model_name= 'dense'
# factor=2
# checkpoint='outputs/resolution_dataset25/srdense/dense_downsample_z_axis25_mask_training_addition_f2_105_0.0001/checkpoints/z_axis/factor_2/epoch_80_f_2.pth'
# plot_dir='test_set/plots_2/'
# preds_dir='test_set/preds_2/'
# edge_type='downsample'
# pred_edges_dir='test_set/preds_edges_2/'
# input_edges_dir='test_set/input_edges_2/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")


# # Downsample Addition factor 4 (Done)
# model_name= 'dense'
# factor=4
# checkpoint = 'outputs/resolution_dataset25/srdense/dense_downsample_z_axis25_mask_training_addition_f4_805_0.0001/checkpoints/z_axis/factor_4/epoch_100_f_4.pth'
# plot_dir=f'test_set/plots_{factor}/'
# preds_dir=f'test_set/preds_{factor}/'
# edge_type='downsample'
# pred_edges_dir=f'test_set/preds_edges_{factor}/'
# input_edges_dir=f'test_set/input_edges_{factor}/'
# os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")
