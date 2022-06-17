import os


os.system("python test.py --checkpoint='outputs/resolution_dataset50/srdense/SRDENSE_GR7NB5NL5S4_KAI_NE3005_BS32_LR0.006_Z_F4_ADAM_MSE/checkpoints/z_axis/factor_4/best_weights_factor_4_epoch_12.pth' --model_name='dense' --factor=2 --plot-dir='test_set/plots_4/' --preds-dir='test_set/preds_4/'")