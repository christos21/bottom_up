import os
import pandas as pd
import tensorflow as tf

from components.smart_grid import SmartGrid

base_folder = 'final_scenarios/scenario_3/config_files'

grid_csv            = os.path.join(base_folder, 'grid_IEEE.csv')
simulation_file     = os.path.join(base_folder, 'simulation_parameters.csv')
generators_csv      = os.path.join(base_folder, 'generators.csv')
dss_file            = os.path.join(base_folder, 'IEEE.dss')
pv_battery_csv      = os.path.join(base_folder, 'pv_battery.csv')

ev_model = 'final_scenarios/scenario_3/EV_model.hdf5'

# initialize grid
grid = SmartGrid(csv_file=grid_csv, name='IEEE')
grid.set_simulation_parameters(simulation_file)
grid.set_load_consumption()

df = pd.read_csv(pv_battery_csv, index_col=0)
df_gen = pd.read_csv(generators_csv, index_col=0)

grid.reset_pv_and_battery(df)
grid.set_generators(df_gen)

# load EV model
model = tf.keras.models.load_model(ev_model)

# parameters for NILM model
n_steps = 5*60
n_features = 1
mains_factor = 10000
gt_factor = 8000

# Set the following value to find EVs only after 17:00
time_int = 6120 # 17:00

nodes_with_ev = []

# Test all end-user nodes
for node in df.columns:
    # get the power and find EV usae
    power = grid[node].P.sum(axis=1)
    samples = power.to_numpy().reshape(-1, n_steps)/mains_factor
    pred = model.predict(samples) * gt_factor
    pred = pred.reshape(-1)

    # if predicted power after 17:00 is significant there is an EV usage
    if max(pred[time_int//(5*6):]) > 2000:
        nodes_with_ev.append(node)

print(nodes_with_ev)

"""
The list of identified nodes with EV after 17:00
['b7', 'b20', 'b31', 'b39', 'b41', 'b45', 'b46', 'b40', 'b49', 'b50', 'b55', 'b56', 'b64', 'b61', 'b103', 'b113', 
 'b112', 'b97', 'b141', 'b89', 'b149', 'b154', 'b166', 'b172', 'b170', 'b192', 'b199', 'b207', 'b205', 'b204']

"""
