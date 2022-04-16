from components.smart_grid import SmartGrid

grid_csv        = 'test_scripts/grid_IEEE.csv'
simulation_file = 'final_scenarios/scenario_1/config_files/simulation_parameters.csv'
solar_parks_csv = 'final_scenarios/scenario_1/config_files/generators.csv'
dss_file        = 'final_scenarios/scenario_1/config_files/IEEE.dss'
line_limits     = 'final_scenarios/line_limits.csv'

results_folder  = 'demo'

# Initialize the grid
grid = SmartGrid(csv_file=grid_csv, name='IEEE')

# Set the simulation parameters and calculate the power for each home.
grid.set_simulation_parameters(simulation_file)
grid.set_load_consumption()

# Add solar panels at free nodes of the grid.
grid.set_generators(csv_file=solar_parks_csv)


# Solve power flow for 5 minutes (starting at 1000th second) at a 10 sec interval
# and save the results to the results folder

grid.solve_power_flow(dss_path=dss_file,
                      number_of_seconds=5*60,
                      starting_second=1000,
                      subsampling=10,
                      folder=results_folder)

# calculate grid indexes
grid.calculate_grid_indexes(power_flow_results=results_folder,
                            grid_params=grid.grid_parameters(dss_file),
                            line_limits_file=line_limits)

