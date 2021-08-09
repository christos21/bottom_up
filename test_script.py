# demo script
from smart_grid import SmartGrid

# Initialize the grid. In this case, there are no PV and batteries. By editing the corresponding cells
# of the csv file, you can add PV and battery to whichever home you prefer.
grid = SmartGrid(csv_file='test_cases/test_case_1_minor_over_voltage/grid_IEEE.csv', name='IEEE')

# Set the simulation parameters and calculate the power for each home.
simulation_file = 'test_cases/test_case_1_minor_over_voltage/simulation_parameters.csv'
grid.set_simulation_parameters(simulation_file)
grid.set_load_consumption()

# Add solar panels at free nodes of the grid.
grid.set_generators(csv_file='test_cases/test_case_1_minor_over_voltage/generators.csv')

# Solve power flow for 30 seconds (starting at 1000th second) and save the results to the folder 'demo'
grid.solve_power_flow(dss_path='test_cases/test_case_1_minor_over_voltage/IEEE.dss', number_of_seconds=30,
                      starting_second=1000, save_results=True, folder='demo')

# calculate grid indexes
grid.calculate_grid_indexes()
