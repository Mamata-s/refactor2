import os



# Downsample Original factor 2 (Done)
model_name= 'dense'
factor = 2
checkpoint='colab_checkpoints/checkpoint_gaussian_epoch_400.pth'
plot_dir= f'test_set/plots_{factor}/'
preds_dir=f'test_set/preds_{factor}/'
edge_type='downsample'
pred_edges_dir=f'test_set/preds_edges_{factor}/'
input_edges_dir=f'test_set/input_edges_{factor}/'
os.system(f"python test1.py --checkpoint={checkpoint} --model_name={model_name} --factor={factor} --plot-dir={plot_dir} --preds-dir={preds_dir} --edges --edge-type={edge_type} --pred-edges-dir={pred_edges_dir} --input-edges-dir={input_edges_dir}")
