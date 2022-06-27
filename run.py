import os

# DONE
# # ******* Get result of  downsample edges with losstype original ie loss calculate on edges
model_name= 'dense'
factor=2
checkpoint='outputs/resolution_dataset50/srdense/SRDENSE_GR7NB5NL5S4_DOWNSAMPLE_EDGES_MSE_ADAM_F2_LR0.001_FULL50_EP5005_initNorm/checkpoints/full/factor_2/epoch_5000_f_2.pth'
plot_dir='test_set/plots_2/'
preds_dir='test_set/preds_2/'
edge_type='downsample'
pred_edges_dir='test_set/preds_edges_2/'
input_edges_dir='test_set/input_edges_2/'
os.system(f"python test.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")



# NEED TO DEBUG
# ******* Get result of  downsample edges with losstype ADDITION ie loss calculate on edges
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


