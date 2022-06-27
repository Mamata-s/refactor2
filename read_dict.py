import pickle as pkl
import yaml
# reading the data from the file
# pkl_path ='outputs/resolution_dataset50/srdense/srdense_edge_training_4/losses/z_axis/factor_4/configuration'
pkl_path='outputs/resolution_dataset50/srdense/srdense_edge_ltype_addition_4/losses/z_axis/factor_4/configuration'
with open(pkl_path, 'rb') as f:
    data = pkl.load(f)


# print(data)
with open('config.yml', 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)