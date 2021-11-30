import os

import opendssdirect as dss
import time
import numpy as np
import pandas as pd
import json

from utils import PHASES_INV

# nominal power of transformer
TRF_NOM_POWER = 250


def voltage_extraction():
    """
    Extract voltages in each power flow step.
    :return: np.array
    """

    node_names = dss.Circuit.AllNodeNames()
    bus_names = dss.Circuit.AllBusNames()
    buss_voltages = dss.Circuit.AllBusVolts()

    lb = len(bus_names)
    ln = len(node_names)

    VNn = np.zeros(shape=(lb, 1), dtype=np.complex)
    VAn = np.zeros(shape=(lb, 1), dtype=np.complex)
    VBn = np.zeros(shape=(lb, 1), dtype=np.complex)
    VCn = np.zeros(shape=(lb, 1), dtype=np.complex)
    Voltage = np.zeros(shape=(lb, 3), dtype=np.complex)

    for x in range(0, lb):
        node1 = bus_names[x] + '.1'
        node2 = bus_names[x] + '.2'
        node3 = bus_names[x] + '.3'
        node4 = bus_names[x] + '.4'

        for j in range(0, ln):
            if node1 == node_names[j]:
                VAn[x] = buss_voltages[2 * j] + buss_voltages[2 * j + 1] * 1j
            elif node2 == node_names[j]:
                VBn[x] = buss_voltages[2 * j] + buss_voltages[2 * j + 1] * 1j
            elif node3 == node_names[j]:
                VCn[x] = buss_voltages[2 * j] + buss_voltages[2 * j + 1] * 1j
            elif node4 == node_names[j]:
                VNn[x] = buss_voltages[2 * j] + buss_voltages[2 * j + 1] * 1j

    for x in range(0, lb):
        Voltage[x, 0] = VAn[x] - VNn[x]
        Voltage[x, 1] = VBn[x] - VNn[x]
        Voltage[x, 2] = VCn[x] - VNn[x]

    return Voltage


def current_extraction():
    """
    Extract currents (phases a, b, c and neutral) in each power flow step.
    :return: np.array
    :return:
    """
    ln = len(dss.Lines.AllNames())

    current = np.zeros(shape=(ln, 4), dtype=np.complex)

    for k, _ in enumerate(dss.Lines.AllNames()):
        dss.Lines.Idx(k + 1)
        temp = dss.CktElement.Currents()

        current[k, 0] = temp[0] + temp[1] * 1j  # Phase a
        current[k, 1] = temp[2] + temp[3] * 1j  # Phase b
        current[k, 2] = temp[4] + temp[5] * 1j  # Phase c
        current[k, 3] = temp[6] + temp[7] * 1j  # Neutral

    return current


def get_reactive_from_active_and_power_factor(p, pf):
    return (p ** 2 / pf ** 2 - p ** 2)**(1/2)


def solve_power_flow(grid, dss_path='test_grids/JRC.dss', number_of_seconds=None, starting_second=0,
                     save_results=False, folder=None, subsampling=1, home_power_factor=None):
    """
    Function for solving the power flow.
    :param grid: SmartGrid
    :param dss_path: str, Path to a .dss file. The name of each home should be the name
                          of the bus that is connected to.
    :param number_of_seconds: int, Number of seconds to solve the power flow
    :param starting_second: int, The first second for which the power flow will be solved.
    :param save_results: bool, Set to True to save results.
    :param folder: str, Path of the folder where the results will be saved if save_results is True.

    :return: (str, str) if save_results is True,
            (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame) otherwise
    """

    # run dss file to load circuit, bus names, lines etc
    dss.run_command('Redirect ' + dss_path)
    dss.run_command('solve')

    # Create list of loads
    load_names = list(grid.homes.keys())
    for load_name in load_names:
        assert load_name in dss.Circuit.AllBusNames(), 'Home {} does not correspond to a bus. Each home should ' \
                                                       'have the name of the bus that is attached on.'.format(load_name)

    # Create list of solar panels at free nodes
    generator_names = list(grid.generators.keys())
    for generator_name in generator_names:
        assert generator_name in dss.Circuit.AllBusNames(), 'Generator {} does not correspond to a bus. ' \
                                                            'Each generator should have the name of the bus that is ' \
                                                            'attached on.'.format(generator_name)

    # Calculation of the execution time
    start = time.time()

    # Acquire the length of the time instants
    if number_of_seconds:
        tm = number_of_seconds
    else:
        tm = len(grid.P)

    assert starting_second < len(grid.P), 'Starting second > simulation period'

    ending_second = min(starting_second + tm, len(grid.P))

    tm = ending_second - starting_second

    df_index = grid.P.index[starting_second:ending_second:subsampling]

    # Initialize lists and arrays
    Va, Vb, Vc = [], [], []
    Ca, Cb, Cc, Cn = [], [], [], []

    Va_vec, Vb_vec, Vc_vec = [], [], []

    array_dim = len(df_index)

    Losses = np.zeros((array_dim, 2))
    S_trf = np.zeros(shape=(array_dim, 1), dtype=np.complex)
    S_trf_sec = np.zeros(shape=(array_dim, 1), dtype=np.complex)

    line_power = np.zeros(shape=(array_dim, len(dss.Lines.AllNames())), dtype=np.complex)

    # loop for each second of simulation
    for t_abs, t in enumerate(range(0, tm, subsampling)):
        if t % (60*60*6) == 0:
            print("Time-step #{} from {}".format(t, tm))

        dss.run_command('Redirect ' + dss_path)

        # Set all the loads
        for i, load in enumerate(load_names):
            if grid.homes[load].single_phase:
                phase_id = PHASES_INV[grid.homes[load].selected_phase]

                if home_power_factor:
                    p = grid.homes[load].P.iloc[t + starting_second, phase_id] / 1000
                    kvar = get_reactive_from_active_and_power_factor(p, home_power_factor)
                else:
                    kvar = grid.homes[load].Q.iloc[t + starting_second, phase_id] / 1000

                cmd = "New Load.{load_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 phases=1 kV=(0.416 3 sqrt /) " \
                      "kW={kW} kvar={kvar} model=1 status=fixed  Vminpu=0.1 Vmaxpu=2".format(
                        load_name='Load_' + load,
                        phase_name=grid.homes[load].selected_phase,
                        bus_name=load,
                        bus_phase=phase_id+1,
                        kW=-grid.homes[load].grid_power.iloc[t+starting_second, phase_id]/1000,
                        kvar=kvar
                      )
                dss.run_command(cmd)
            else:
                phases = ['a', 'b', 'c']
                for k, ph in enumerate(phases):

                    if home_power_factor:
                        p = grid.homes[load].P.iloc[t+starting_second, k]/1000
                        kvar = get_reactive_from_active_and_power_factor(p, home_power_factor)
                    else:
                        kvar = grid.homes[load].Q.iloc[t+starting_second, k]/1000

                    cmd = "New Load.{load_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 phases=1 kV=(0.416 3 sqrt /) " \
                          "kW={kW} kvar={kvar} model=1 status=fixed  Vminpu=0.1 Vmaxpu=2".format(
                            load_name='Load_' + load,
                            phase_name=ph,
                            bus_name=load,
                            bus_phase=k+1,
                            kW=-grid.homes[load].grid_power.iloc[t+starting_second, k]/1000,
                            kvar=kvar
                        )
                    dss.run_command(cmd)

        # set all generators at free nodes
        for i, generator in enumerate(generator_names):
            if grid.generators[generator].single_phase:
                phase_id = PHASES_INV[grid.generators[generator].selected_phase]
                dss.run_command(
                    "New generator.{generator_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 " \
                    "phases=1 kV=(0.416 3 sqrt /) " \
                    "kW={kW} kvar={kvar} model=1 status=fixed  Vmin=0.1 Vmax=2".format(
                        generator_name='PV_' + generator,
                        phase_name=grid.generators[generator].selected_phase,
                        bus_name=generator,
                        bus_phase=phase_id+1,
                        kW=grid.generators[generator].P.iloc[t+starting_second, phase_id]/1000,
                        kvar=0
                    ))
            else:
                phases = ['a', 'b', 'c']
                for k, ph in enumerate(phases):
                    dss.run_command(
                        "New generator.{generator_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 " \
                        "phases=1 kV=(0.416 3 sqrt /) " \
                        "kW={kW} kvar={kvar} model=1 status=fixed  Vmin=0.1 Vmax=2".format(
                            generator_name='PV_' + generator,
                            phase_name=ph,
                            bus_name=generator,
                            bus_phase=k+1,
                            kW=grid.generators[generator].P.iloc[t+starting_second, k]/1000,
                            kvar=0
                        ))

        # solve power flow for the current second
        dss.run_command('Solve')

        # Check for convergence
        converg = dss.Solution.Converged()
        if not converg:
            print("Solution did not converge at time instant ", grid.P.index[t])

        # calculate losses
        Losses[t_abs] = dss.Circuit.Losses()

        # Calculate network voltages
        Voltage = voltage_extraction()
        Volt_abs = abs(Voltage)

        # calculate currents
        Current = current_extraction()
        Curr_abs = abs(Current)

        # calculate power in each line
        for k, line_ in enumerate(dss.Lines.AllNames()):
            dss.Lines.Idx(k+1)
            temp = dss.CktElement.Powers()
            temp_1 = temp[:int(len(temp)/2)]
            temp_2 = temp[int(len(temp) / 2):]
            temp_sum = np.array(temp_1)  # + np.array(temp_2)
            p_, q_ = sum(temp_sum[:7:2]), sum(temp_sum[1:8:2])
            line_power[t_abs, k] = p_ + q_*1j  # + sum(temp_2[:7:2]) + sum(temp_sum[1:8:2])*1j

        # calculate transformer power
        for k, transformer in enumerate(dss.Transformers.AllNames()):
            dss.Transformers.Idx(k+1)
            temp = dss.CktElement.Powers()
            S_trf[t_abs] += temp[0] + temp[1] * 1j
            S_trf_sec[t_abs] += temp[4] + temp[5] * 1j

        # append the results of this second to the lists
        Va_vec.append(Voltage[:, 0])
        Vb_vec.append(Voltage[:, 1])
        Vc_vec.append(Voltage[:, 2])

        Va.append(Volt_abs[:, 0])
        Vb.append(Volt_abs[:, 1])
        Vc.append(Volt_abs[:, 2])
        Ca.append(Curr_abs[:, 0])
        Cb.append(Curr_abs[:, 1])
        Cc.append(Curr_abs[:, 2])
        Cn.append(Curr_abs[:, 3])

    # Store results as dataframes
    df_index = grid.P.index[starting_second:ending_second:subsampling]

    Va = pd.DataFrame(index=df_index, data=np.array(Va), columns=dss.Circuit.AllBusNames())
    Vb = pd.DataFrame(index=df_index, data=np.array(Vb), columns=dss.Circuit.AllBusNames())
    Vc = pd.DataFrame(index=df_index, data=np.array(Vc), columns=dss.Circuit.AllBusNames())

    Va_vec = pd.DataFrame(index=df_index, data=np.array(Va_vec), columns=dss.Circuit.AllBusNames())
    Vb_vec = pd.DataFrame(index=df_index, data=np.array(Vb_vec), columns=dss.Circuit.AllBusNames())
    Vc_vec = pd.DataFrame(index=df_index, data=np.array(Vc_vec), columns=dss.Circuit.AllBusNames())

    Ca = pd.DataFrame(index=df_index, data=np.array(Ca), columns=dss.Lines.AllNames())
    Cb = pd.DataFrame(index=df_index, data=np.array(Cb), columns=dss.Lines.AllNames())
    Cc = pd.DataFrame(index=df_index, data=np.array(Cc), columns=dss.Lines.AllNames())
    Cn = pd.DataFrame(index=df_index, data=np.array(Cn), columns=dss.Lines.AllNames())

    # Calculate transformer total real and reactive power
    P_f = S_trf.real
    transformer_power = pd.DataFrame(index=df_index,
                                     data=np.array([S_trf.real.reshape(-1), S_trf.imag.reshape(-1),
                                                    S_trf_sec.real.reshape(-1), S_trf_sec.imag.reshape(-1)]).T,
                                     columns=['P_f', 'Q_f', 'P_s', 'Q_s'])

    # Calculate energy losses (in kWh) assuming 1s time-resolution
    Energy_Loss = np.sum(Losses[:, 0]) / (1000*3600/subsampling)

    Losses = pd.DataFrame(index=df_index, data=Losses, columns=['P', 'Q'])

    line_power_df = pd.DataFrame(index=df_index, columns=dss.Lines.AllNames(), data=abs(line_power))

    end = time.time()

    print('\n|||------------------------- Simulation summary -------------------------|||\n')
    print('Simulation period = ', tm, 's')
    print('Elapsed time =', end - start, ' s')
    print('Network losses =', Energy_Loss, ' kWh')

    write_flag = False
    if save_results and os.path.exists(folder):
        user_input = input('Folder already exist. Do you want to rewrite? [Y/N] \n')
        if user_input.lower() == 'y':
            write_flag = True
    elif save_results:
        os.makedirs(folder)
        write_flag = True

    if write_flag and save_results:
        # save power flow results
        Va.to_csv(os.path.join(folder, 'Va.csv'))
        Vb.to_csv(os.path.join(folder, 'Vb.csv'))
        Vc.to_csv(os.path.join(folder, 'Vc.csv'))

        Va_vec.to_csv(os.path.join(folder, 'Va_vec.csv'))
        Vb_vec.to_csv(os.path.join(folder, 'Vb_vec.csv'))
        Vc_vec.to_csv(os.path.join(folder, 'Vc_vec.csv'))

        Ca.to_csv(os.path.join(folder, 'Ca.csv'))
        Cb.to_csv(os.path.join(folder, 'Cb.csv'))
        Cc.to_csv(os.path.join(folder, 'Cc.csv'))
        Cn.to_csv(os.path.join(folder, 'Cn.csv'))

        Losses.to_csv(os.path.join(folder, 'Losses.csv'))
        transformer_power.to_csv(os.path.join(folder, 'transformer_power.csv'))

        line_power_df.to_csv(os.path.join(folder, 'line_power.csv'))

    else:

        return Va_vec, Vb_vec, Vc_vec, Ca, Cb, Cc, Cn


def calculate_passive_network_losses_from_power_flow(grid, dss_path='test_grids/JRC.dss',
                                                     number_of_seconds=None, starting_second=0,
                                                     save_results=False, folder=None, subsampling=1,
                                                     home_power_factor=None):
    """
    Function for solving the power flow without considering PV and BESS in order to calculate losses
    for passive network.
    :param grid: SmartGrid
    :param dss_path: str, Path to a .dss file. The name of each home should be the name
                          of the bus that is connected to.
    :param number_of_seconds: int, Number of seconds to solve the power flow
    :param starting_second: int, The first second for which the power flow will be solved.
    :param save_results: bool, Set to True to save results.
    :param folder: str, Path of the folder where the results will be saved if save_results is True.

    :return: (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame) if save_results is False
    """

    # run dss file to load circuit, bus names, lines etc
    dss.run_command('Redirect ' + dss_path)
    dss.run_command('solve')

    # Create list of loads
    load_names = list(grid.homes.keys())
    for load_name in load_names:
        assert load_name in dss.Circuit.AllBusNames(), 'Home {} does not correspond to a bus. Each home should ' \
                                                       'have the name of the bus that is attached on.'.format(load_name)

    # Calculation of the execution time
    start = time.time()

    # Acquire the length of the time instants
    if number_of_seconds:
        tm = number_of_seconds
    else:
        tm = len(grid.P)

    assert starting_second < len(grid.P), 'Starting second > simulation period'

    ending_second = min(starting_second + tm, len(grid.P))

    tm = ending_second - starting_second

    df_index = grid.P.index[starting_second:ending_second:subsampling]
    array_dim = len(df_index)

    # Initialize lists and arrays
    Va, Vb, Vc = [], [], []

    Va_vec, Vb_vec, Vc_vec = [], [], []

    Losses = np.zeros((array_dim, 2))

    # loop for each second of simulation
    for t_abs, t in enumerate(range(0, tm, subsampling)):
        if t % (60*60*6) == 0:
            print("Time-step #{} from {}".format(t, tm))

        dss.run_command('Redirect ' + dss_path)

        # Set all the loads
        for i, load in enumerate(load_names):
            if grid.homes[load].single_phase:
                phase_id = PHASES_INV[grid.homes[load].selected_phase]

                if home_power_factor:
                    p = grid.homes[load].P.iloc[t+starting_second, phase_id]/1000
                    kvar = get_reactive_from_active_and_power_factor(p, home_power_factor)
                else:
                    kvar = grid.homes[load].Q.iloc[t+starting_second, phase_id]/1000

                cmd = "New Load.{load_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 phases=1 kV=(0.416 3 sqrt /) " \
                      "kW={kW} kvar={kvar} model=1 status=fixed  Vminpu=0.1 Vmaxpu=2".format(
                        load_name='Load_' + load,
                        phase_name=grid.homes[load].selected_phase,
                        bus_name=load,
                        bus_phase=phase_id+1,
                        kW=grid.homes[load].P.iloc[t+starting_second, phase_id]/1000,
                        kvar=kvar
                      )
                dss.run_command(cmd)
            else:
                phases = ['a', 'b', 'c']
                for k, ph in enumerate(phases):

                    if home_power_factor:
                        p = grid.homes[load].P.iloc[t+starting_second, k]/1000
                        kvar = get_reactive_from_active_and_power_factor(p, home_power_factor)
                    else:
                        kvar = grid.homes[load].Q.iloc[t+starting_second, k]/1000

                    cmd = "New Load.{load_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 phases=1 kV=(0.416 3 sqrt /) " \
                          "kW={kW} kvar={kvar} model=1 status=fixed  Vminpu=0.1 Vmaxpu=2".format(
                            load_name='Load_' + load,
                            phase_name=ph,
                            bus_name=load,
                            bus_phase=k+1,
                            kW=grid.homes[load].P.iloc[t+starting_second, k]/1000,
                            kvar=kvar
                        )
                    dss.run_command(cmd)

        # solve power flow for the current second
        dss.run_command('Solve')

        # Check for convergence
        converg = dss.Solution.Converged()
        if not converg:
            print("Solution did not converge at time instant ", grid.P.index[t])

        # calculate losses
        Losses[t_abs] = dss.Circuit.Losses()

        # Calculate network voltages
        Voltage = voltage_extraction()
        Volt_abs = abs(Voltage)

        # append the results of this second to the lists
        Va_vec.append(Voltage[:, 0])
        Vb_vec.append(Voltage[:, 1])
        Vc_vec.append(Voltage[:, 2])

        Va.append(Volt_abs[:, 0])
        Vb.append(Volt_abs[:, 1])
        Vc.append(Volt_abs[:, 2])

    # Store results as dataframes
    df_index = grid.P.index[starting_second:ending_second:subsampling]

    Va = pd.DataFrame(index=df_index, data=np.array(Va), columns=dss.Circuit.AllBusNames())
    Vb = pd.DataFrame(index=df_index, data=np.array(Vb), columns=dss.Circuit.AllBusNames())
    Vc = pd.DataFrame(index=df_index, data=np.array(Vc), columns=dss.Circuit.AllBusNames())

    Va_vec = pd.DataFrame(index=df_index, data=np.array(Va_vec), columns=dss.Circuit.AllBusNames())
    Vb_vec = pd.DataFrame(index=df_index, data=np.array(Vb_vec), columns=dss.Circuit.AllBusNames())
    Vc_vec = pd.DataFrame(index=df_index, data=np.array(Vc_vec), columns=dss.Circuit.AllBusNames())

    Losses = pd.DataFrame(index=df_index, data=Losses, columns=['P', 'Q'])

    end = time.time()

    print('\n|||------------------------- Simulation summary -------------------------|||\n')
    print('Simulation period = ', tm, 's')
    print('Elapsed time =', end - start, ' s')

    if save_results:

        # save power flow results
        Va.to_csv(os.path.join(folder, 'Va_for_passive_network.csv'))
        Vb.to_csv(os.path.join(folder, 'Vb_for_passive_network.csv'))
        Vc.to_csv(os.path.join(folder, 'Vc_for_passive_network.csv'))

        # save power flow results
        Va_vec.to_csv(os.path.join(folder, 'Va_vec_for_passive_network.csv'))
        Vb_vec.to_csv(os.path.join(folder, 'Vb_vec_for_passive_network.csv'))
        Vc_vec.to_csv(os.path.join(folder, 'Vc_vec_for_passive_network.csv'))

        Losses.to_csv(os.path.join(folder, 'Losses_for_passive_network.csv'))

    else:

        return Losses, Va_vec, Vb_vec, Vc_vec


def get_grid_info_from_dss(dss_path, folder=None, save=False):
    # run dss file to load circuit, bus names, lines etc
    dss.run_command('Redirect ' + dss_path)
    dss.run_command('solve')

    # Calculate grid parameters and save them to json file
    # Apparent power for the substation
    S_substation = 0
    for k, transformer in enumerate(dss.Transformers.AllNames()):
        dss.Transformers.Idx(k + 1)
        temp = dss.Transformers.kVA()
        S_substation += temp

    # transformer secondary terminal bus
    dss.Transformers.Idx(1)
    transformer_1_buses = [b.split('.')[0] for b in dss.CktElement.BusNames()]
    assert len(transformer_1_buses) == 2, 'There should be 2 substation buses'
    initial_bus = transformer_1_buses[1]

    for k, transformer in enumerate(dss.Transformers.AllNames()):
        dss.Transformers.Idx(k + 1)
        transformer_buses = [b.split('.')[0] for b in dss.CktElement.BusNames()]
        assert len(transformer_buses) == 2, 'There should be 2 substation buses'
        assert initial_bus == transformer_buses[1]

    # line length
    line_lengths = {}

    line_lengths_using_bus_names = {}

    line_resistance_using_bus_names = {}

    total_length = 0
    for k, line_ in enumerate(dss.Lines.AllNames()):
        dss.Lines.Idx(k + 1)
        line_lengths[line_] = dss.Lines.Length()
        total_length += line_lengths[line_]

        bus1 = dss.Lines.Bus1().split('.')[0]
        bus2 = dss.Lines.Bus2().split('.')[0]
        line_lengths_using_bus_names['{}-{}'.format(bus1, bus2)] = dss.Lines.Length()
        line_resistance_using_bus_names['{}-{}'.format(bus1, bus2)] = dss.Lines.Length()*1000*dss.Lines.R1()

    # bus names
    bus_names = [bus for bus in dss.Circuit.AllBusNames()]

    # bus connections
    bus_connections = {}
    for bus in bus_names:
        connections = []
        for k, _ in enumerate(dss.Lines.AllNames()):
            dss.Lines.Idx(k + 1)
            bus1 = dss.Lines.Bus1().split('.')[0]
            bus2 = dss.Lines.Bus2().split('.')[0]
            if bus == bus1:
                connections.append(bus2)
            elif bus == bus2:
                connections.append(bus1)

        bus_connections[bus] = connections

        # if bus in load_names:
        #     assert len(connections) == 1, 'House {} should be connected to a single bus.'.format(bus)
        # else:
        #     bus_connections[bus] = connections

    length_from_start_to_each_bus = calculate_attribute_from_start_to_each_bus(
        bus_names, line_lengths_using_bus_names, bus_connections, initial_bus)

    resistance_from_start_to_each_bus = calculate_attribute_from_start_to_each_bus(
        bus_names, line_resistance_using_bus_names, bus_connections, initial_bus)

    param_dict = {'folder': folder,
                  'S_substation': S_substation,
                  'initial_bus': initial_bus,
                  'line_length': line_lengths,
                  'total_length': total_length,
                  'bus_names': bus_names,
                  'bus_connections': bus_connections,
                  'length_from_start_to_each_bus': length_from_start_to_each_bus,
                  'resistance_from_start_to_each_bus': resistance_from_start_to_each_bus}

    if save:
        with open(os.path.join(folder, 'parameters.json'), 'w') as fp:
            json.dump(param_dict, fp, sort_keys=True, indent=4)
    else:
        return param_dict


def calculate_attribute_from_start_to_each_bus(bus_names, line_attribute_using_bus_names, bus_connections, initial_bus):

    attribute_from_start = {bus: None for bus in bus_names}

    attribute_from_start[initial_bus] = 0

    def calculate_from_this_bus(current_bus):

        connections = bus_connections[current_bus]


        for conn in connections:
            if attribute_from_start[conn] is None:

                attribute_from_start[conn] = attribute_from_start[current_bus] + line_attribute_using_bus_names[
                    '{}-{}'.format(current_bus, conn)]

                calculate_from_this_bus(conn)

    calculate_from_this_bus(initial_bus)

    return attribute_from_start


def solve_power_flow_with_single_load(grid, bus, dss_path='test_grids/JRC.dss', number_of_seconds=None, starting_second=0,
                     save_results=False, folder=None, subsampling=1, home_power_factor=None):
    """
    Function for solving the power flow.
    :param grid: SmartGrid
    :param dss_path: str, Path to a .dss file. The name of each home should be the name
                          of the bus that is connected to.
    :param number_of_seconds: int, Number of seconds to solve the power flow
    :param starting_second: int, The first second for which the power flow will be solved.
    :param save_results: bool, Set to True to save results.
    :param folder: str, Path of the folder where the results will be saved if save_results is True.

    :return: (str, str) if save_results is True,
            (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame) otherwise
    """

    # run dss file to load circuit, bus names, lines etc
    dss.run_command('Redirect ' + dss_path)
    dss.run_command('solve')

    # Create list of loads
    load_names = list(grid.homes.keys())
    assert bus in load_names
    for load_name in load_names:
        assert load_name in dss.Circuit.AllBusNames(), 'Home {} does not correspond to a bus. Each home should ' \
                                                       'have the name of the bus that is attached on.'.format(load_name)

    # Calculation of the execution time
    start = time.time()

    # Acquire the length of the time instants
    if number_of_seconds:
        tm = number_of_seconds
    else:
        tm = len(grid.P)

    assert starting_second < len(grid.P), 'Starting second > simulation period'

    ending_second = min(starting_second + tm, len(grid.P))

    tm = ending_second - starting_second

    df_index = grid.P.index[starting_second:ending_second:subsampling]

    # Initialize lists and arrays
    Va, Vb, Vc = [], [], []
    Ca, Cb, Cc, Cn = [], [], [], []

    Va_vec, Vb_vec, Vc_vec = [], [], []

    array_dim = len(df_index)

    Losses = np.zeros((array_dim, 2))
    S_trf = np.zeros(shape=(array_dim, 1), dtype=np.complex)
    S_trf_sec = np.zeros(shape=(array_dim, 1), dtype=np.complex)

    line_power = np.zeros(shape=(array_dim, len(dss.Lines.AllNames())), dtype=np.complex)

    # loop for each second of simulation
    for t_abs, t in enumerate(range(0, tm, subsampling)):
        if t % (60*60*6) == 0:
            print("Time-step #{} from {}".format(t, tm))

        dss.run_command('Redirect ' + dss_path)

        # Set all the loads
        load = bus
        i = load_names.index(bus)

        if grid.homes[load].single_phase:
            phase_id = PHASES_INV[grid.homes[load].selected_phase]

            if home_power_factor:
                p = grid.homes[load].P.iloc[t + starting_second, phase_id] / 1000
                kvar = get_reactive_from_active_and_power_factor(p, home_power_factor)
            else:
                kvar = grid.homes[load].Q.iloc[t + starting_second, phase_id] / 1000

            cmd = "New Load.{load_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 phases=1 kV=(0.416 3 sqrt /) " \
                  "kW={kW} kvar={kvar} model=1 status=fixed  Vminpu=0.1 Vmaxpu=2".format(
                    load_name='Load_' + load,
                    phase_name=grid.homes[load].selected_phase,
                    bus_name=load,
                    bus_phase=phase_id+1,
                    kW=-grid.homes[load].grid_power.iloc[t+starting_second, phase_id]/1000,
                    kvar=kvar
                  )
            dss.run_command(cmd)
        else:
            phases = ['a', 'b', 'c']
            for k, ph in enumerate(phases):

                if home_power_factor:
                    p = grid.homes[load].P.iloc[t+starting_second, k]/1000
                    kvar = get_reactive_from_active_and_power_factor(p, home_power_factor)
                else:
                    kvar = grid.homes[load].Q.iloc[t+starting_second, k]/1000

                cmd = "New Load.{load_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 phases=1 kV=(0.416 3 sqrt /) " \
                      "kW={kW} kvar={kvar} model=1 status=fixed  Vminpu=0.1 Vmaxpu=2".format(
                        load_name='Load_' + load,
                        phase_name=ph,
                        bus_name=load,
                        bus_phase=k+1,
                        kW=-grid.homes[load].grid_power.iloc[t+starting_second, k]/1000,
                        kvar=kvar
                    )
                dss.run_command(cmd)

        # solve power flow for the current second
        dss.run_command('Solve')

        # Check for convergence
        converg = dss.Solution.Converged()
        if not converg:
            print("Solution did not converge at time instant ", grid.P.index[t])

        # calculate losses
        Losses[t_abs] = dss.Circuit.Losses()

        # Calculate network voltages
        Voltage = voltage_extraction()
        Volt_abs = abs(Voltage)

        # calculate currents
        Current = current_extraction()
        Curr_abs = abs(Current)

        # calculate power in each line
        for k, line_ in enumerate(dss.Lines.AllNames()):
            dss.Lines.Idx(k+1)
            temp = dss.CktElement.Powers()
            temp_1 = temp[:int(len(temp)/2)]
            temp_2 = temp[int(len(temp) / 2):]
            temp_sum = np.array(temp_1)  # + np.array(temp_2)
            p_, q_ = sum(temp_sum[:7:2]), sum(temp_sum[1:8:2])
            line_power[t_abs, k] = p_ + q_*1j  # + sum(temp_2[:7:2]) + sum(temp_sum[1:8:2])*1j

        # calculate transformer power
        for k, transformer in enumerate(dss.Transformers.AllNames()):
            dss.Transformers.Idx(k+1)
            temp = dss.CktElement.Powers()
            S_trf[t_abs] += temp[0] + temp[1] * 1j
            S_trf_sec[t_abs] += temp[4] + temp[5] * 1j

        # append the results of this second to the lists
        Va_vec.append(Voltage[:, 0])
        Vb_vec.append(Voltage[:, 1])
        Vc_vec.append(Voltage[:, 2])

        Va.append(Volt_abs[:, 0])
        Vb.append(Volt_abs[:, 1])
        Vc.append(Volt_abs[:, 2])
        Ca.append(Curr_abs[:, 0])
        Cb.append(Curr_abs[:, 1])
        Cc.append(Curr_abs[:, 2])
        Cn.append(Curr_abs[:, 3])

    # Store results as dataframes
    df_index = grid.P.index[starting_second:ending_second:subsampling]

    Va = pd.DataFrame(index=df_index, data=np.array(Va), columns=dss.Circuit.AllBusNames())
    Vb = pd.DataFrame(index=df_index, data=np.array(Vb), columns=dss.Circuit.AllBusNames())
    Vc = pd.DataFrame(index=df_index, data=np.array(Vc), columns=dss.Circuit.AllBusNames())

    Va_vec = pd.DataFrame(index=df_index, data=np.array(Va_vec), columns=dss.Circuit.AllBusNames())
    Vb_vec = pd.DataFrame(index=df_index, data=np.array(Vb_vec), columns=dss.Circuit.AllBusNames())
    Vc_vec = pd.DataFrame(index=df_index, data=np.array(Vc_vec), columns=dss.Circuit.AllBusNames())

    Ca = pd.DataFrame(index=df_index, data=np.array(Ca), columns=dss.Lines.AllNames())
    Cb = pd.DataFrame(index=df_index, data=np.array(Cb), columns=dss.Lines.AllNames())
    Cc = pd.DataFrame(index=df_index, data=np.array(Cc), columns=dss.Lines.AllNames())
    Cn = pd.DataFrame(index=df_index, data=np.array(Cn), columns=dss.Lines.AllNames())

    # Calculate transformer total real and reactive power
    P_f = S_trf.real
    transformer_power = pd.DataFrame(index=df_index,
                                     data=np.array([S_trf.real.reshape(-1), S_trf.imag.reshape(-1),
                                                    S_trf_sec.real.reshape(-1), S_trf_sec.imag.reshape(-1)]).T,
                                     columns=['P_f', 'Q_f', 'P_s', 'Q_s'])

    # Calculate energy losses (in kWh) assuming 1s time-resolution
    Energy_Loss = np.sum(Losses[:, 0]) / (1000*3600/subsampling)

    Losses = pd.DataFrame(index=df_index, data=Losses, columns=['P', 'Q'])

    line_power_df = pd.DataFrame(index=df_index, columns=dss.Lines.AllNames(), data=abs(line_power))

    end = time.time()

    print('\n|||------------------------- Simulation summary -------------------------|||\n')
    print('Simulation period = ', tm, 's')
    print('Elapsed time =', end - start, ' s')
    print('Network losses =', Energy_Loss, ' kWh')

    write_flag = False
    if os.path.exists(folder):
        user_input = input('Folder already exist. Do you want to rewrite? [Y/N] \n')
        if user_input.lower() == 'y':
            write_flag = True
    else:
        os.makedirs(folder)
        write_flag = True

    if write_flag and save_results:
        # save power flow results
        Va.to_csv(os.path.join(folder, 'Va.csv'))
        Vb.to_csv(os.path.join(folder, 'Vb.csv'))
        Vc.to_csv(os.path.join(folder, 'Vc.csv'))

        Va_vec.to_csv(os.path.join(folder, 'Va_vec.csv'))
        Vb_vec.to_csv(os.path.join(folder, 'Vb_vec.csv'))
        Vc_vec.to_csv(os.path.join(folder, 'Vc_vec.csv'))

        Ca.to_csv(os.path.join(folder, 'Ca.csv'))
        Cb.to_csv(os.path.join(folder, 'Cb.csv'))
        Cc.to_csv(os.path.join(folder, 'Cc.csv'))
        Cn.to_csv(os.path.join(folder, 'Cn.csv'))

        Losses.to_csv(os.path.join(folder, 'Losses.csv'))
        transformer_power.to_csv(os.path.join(folder, 'transformer_power.csv'))

        line_power_df.to_csv(os.path.join(folder, 'line_power.csv'))

    else:

        return Va_vec, Vb_vec, Vc_vec, Ca, Cb, Cc, Cn
