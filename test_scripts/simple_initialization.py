from components.smart_grid import SmartGrid

grid_csv = 'final_scenarios/scenario_1/config_files/grid_IEEE.csv'
simulation_file = 'final_scenarios/scenario_1/config_files/simulation_parameters.csv'
solar_parks_csv = 'final_scenarios/scenario_1/config_files/generators.csv'

# Initialize the grid
grid = SmartGrid(csv_file=grid_csv, name='IEEE')

# Set the simulation parameters and calculate the power for each home.
grid.set_simulation_parameters(simulation_file)
grid.set_load_consumption()

# Add solar panels at free nodes of the grid.
grid.set_generators(csv_file=solar_parks_csv)

# plot total grid power
grid.P.plot()
