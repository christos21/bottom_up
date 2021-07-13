from smart_grid import SmartGrid


# grid = SmartGrid('test_grids/grid_JRC.csv')
# grid = SmartGrid('test_grids/grid_IEEE.csv', name='IEEE')
grid = SmartGrid('test_cases/grid_IEEE.csv', name='IEEE')

simulation_file = 'test_grids/simulation_parameters.csv'
grid.set_simulation_parameters(simulation_file)
grid.set_load_consumption()


# # grid.set_generators('test_grids/generators_JRC.csv')
# grid.set_generators('test_grids/generators_IEEE.csv')


# grid.solve_power_flow(dss_path='test_grids/IEEE.dss', number_of_seconds=30,
#                       save_results=True, folder='13_6_2021')
# #

