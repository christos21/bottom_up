import os

import opendssdirect as dss
import time
import numpy as np
import pandas as pd
import json

from utils import PHASES_INV


def voltage_extraction():

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


def solve_power_flow(grid, dss_path='test_grids/JRC.dss', number_of_seconds=None, starting_second=0,
                     save_results=False, folder=None):

    # run dss file to load circuit, bus names, lines etc
    dss.run_command('Redirect ' + dss_path)
    dss.run_command('solve')

    # Create list of loads
    load_names = list(grid.homes.keys())
    for load_name in load_names:
        assert load_name in dss.Circuit.AllBusNames(), 'Home {} does not correspond to a bus. Each home should ' \
                                                       'have the name of the bus that is attached on.'.format(load_name)

    generator_names = list(grid.generators.keys())
    for generator_name in generator_names:
        assert generator_name in dss.Circuit.AllBusNames(), 'Generator {} does not correspond to a bus. ' \
                                                            'Each generator should have the name of the bus that is ' \
                                                            'attached on.'.format(generator_name)

    trf_nom_power = 250

    # Initial point for the calculation of the execution time
    start = time.time()

    # Acquire the length of the time instants
    if number_of_seconds:
        tm = number_of_seconds
    else:
        tm = len(grid.P)

    assert starting_second < len(grid.P), 'Starting second > simulation period'

    ending_second = min(starting_second + tm, len(grid.P))

    tm = ending_second - starting_second

    # tm = 60*60*1

    # # Acquire the number of the network loads
    # ls = len(load_names)

    # Initial conditions
    Va, Vb, Vc = [], [], []
    Ca, Cb, Cc, Cn = [], [], [], []

    Va_vec, Vb_vec, Vc_vec = [], [], []
    # Ca_per, Cb_per, Cc_per, Cn_per = [], [], [], []

    Losses = np.zeros((tm, 2))
    S_trf = np.zeros(shape=(tm, 1), dtype=np.complex)
    S_trf_sec = np.zeros(shape=(tm, 1), dtype=np.complex)

    # bus_real_power = np.zeros((tm, len(dss.Circuit.AllBusNames())))
    line_power = np.zeros(shape=(tm, len(dss.Lines.AllNames())), dtype=np.complex)

    # for t in range(60*60*3):
    for t in range(0, tm):
        dss.run_command('Redirect ' + dss_path)

        # Set new grid power in each load
        for i, load in enumerate(load_names):
            if grid.homes[load].single_phase:
                phase_id = PHASES_INV[grid.homes[load].selected_phase]
                cmd = "New Load.{load_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 phases=1 kV=(0.416 3 sqrt /) " \
                      "kW={kW} kvar={kvar} model=1 status=fixed  Vminpu=0.1 Vmaxpu=2".format(
                        load_name='Load_' + load,
                        phase_name=grid.homes[load].selected_phase,
                        bus_name=load,
                        bus_phase=phase_id+1,
                        kW=-grid.homes[load].grid_power.iloc[t+starting_second, phase_id]/1000,
                        kvar=grid.homes[load].Q.iloc[t+starting_second, phase_id]/1000
                      )
                dss.run_command(cmd)
            else:
                phases = ['a', 'b', 'c']
                for k, ph in enumerate(phases):
                    cmd = "New Load.{load_name}{phase_name} Bus1={bus_name}.{bus_phase}.4 phases=1 kV=(0.416 3 sqrt /) " \
                          "kW={kW} kvar={kvar} model=1 status=fixed  Vminpu=0.1 Vmaxpu=2".format(
                            load_name='Load_' + load,
                            phase_name=ph,
                            bus_name=load,
                            bus_phase=k+1,
                            kW=-grid.homes[load].grid_power.iloc[t+starting_second, k]/1000,
                            kvar=grid.homes[load].Q.iloc[t+starting_second, k]/1000
                        )
                    dss.run_command(cmd)

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

        dss.run_command('Solve')

        Losses[t] = dss.Circuit.Losses()

        # # Check for convergence
        converg = dss.Solution.Converged()
        if not converg:
            print("Solution did not converge at time instant ", grid.P.index[t])

        # Calculate network voltages
        Voltage = voltage_extraction()
        Volt_abs = abs(Voltage)

        Current = current_extraction()
        Curr_abs = abs(Current)

        for k, line_ in enumerate(dss.Lines.AllNames()):
            dss.Lines.Idx(k+1)
            temp = dss.CktElement.Powers()
            temp_1 = temp[:int(len(temp)/2)]
            temp_2 = temp[int(len(temp) / 2):]
            temp_sum = np.array(temp_1)  # + np.array(temp_2)
            p_, q_ = sum(temp_sum[:7:2]), sum(temp_sum[1:8:2])
            line_power[t, k] = p_ + q_*1j  # + sum(temp_2[:7:2]) + sum(temp_sum[1:8:2])*1j

            # bus1 = dss.Lines.Bus1().split('.')[0]
            # bus2 = dss.Lines.Bus2().split('.')[0]
            # p_bus1 = sum(temp_1[:6:2])
            # p_bus2 = sum(temp_2[:6:2])

            # bus_real_power[t, dss.Circuit.AllBusNames().index(bus1)] += p_bus1
            # bus_real_power[t, dss.Circuit.AllBusNames().index(bus2)] += p_bus2

        for k, transformer in enumerate(dss.Transformers.AllNames()):
            dss.Transformers.Idx(k+1)
            temp = dss.CktElement.Powers()
            # temp has 8 values, why keep only 2?
            S_trf[t] += temp[0] + temp[1] * 1j
            S_trf_sec[t] += temp[4] + temp[5] * 1j

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
        # Ca_per.append(100 * np.divide(Curr_abs[:, 0], phase_nom_current))
        # Cb_per.append(100 * np.divide(Curr_abs[:, 1], phase_nom_current))
        # Cc_per.append(100 * np.divide(Curr_abs[:, 2], phase_nom_current))
        # Cn_per.append(100 * np.divide(Curr_abs[:, 3], neutral_nom_current))

    df_index = grid.P.index[starting_second:ending_second]

    Va = pd.DataFrame(index=df_index, data=np.array(Va), columns=dss.Circuit.AllBusNames())
    Vb = pd.DataFrame(index=df_index, data=np.array(Vb), columns=dss.Circuit.AllBusNames())
    Vc = pd.DataFrame(index=df_index, data=np.array(Vc), columns=dss.Circuit.AllBusNames())
    Ca = pd.DataFrame(index=df_index, data=np.array(Ca), columns=dss.Lines.AllNames())
    Cb = pd.DataFrame(index=df_index, data=np.array(Cb), columns=dss.Lines.AllNames())
    Cc = pd.DataFrame(index=df_index, data=np.array(Cc), columns=dss.Lines.AllNames())
    Cn = pd.DataFrame(index=df_index, data=np.array(Cn), columns=dss.Lines.AllNames())
    # Ca_per = pd.DataFrame(index=df_index, data=np.array(Ca_per), columns=dss.Lines.AllNames())
    # Cb_per = pd.DataFrame(index=df_index, data=np.array(Cb_per), columns=dss.Lines.AllNames())
    # Cc_per = pd.DataFrame(index=df_index, data=np.array(Cc_per), columns=dss.Lines.AllNames())
    # Cn_per = pd.DataFrame(index=df_index, data=np.array(Cn_per), columns=dss.Lines.AllNames())

    P_f = S_trf.real
    transformer_power = pd.DataFrame(index=df_index,
                                     data=np.array([S_trf.real.reshape(-1), S_trf.imag.reshape(-1),
                                                    S_trf_sec.real.reshape(-1), S_trf_sec.imag.reshape(-1)]).T,
                                     columns=['P_f', 'Q_f', 'P_s', 'Q_s'])
    # originally it was Losses[;,1] but that is reactive, right?
    Energy_Loss = np.sum(Losses[:, 0]) / 3600000  # Calculate energy losses (in kWh) assuming 1s time-resolution
    Losses = pd.DataFrame(index=df_index, data=Losses, columns=['P', 'Q'])

    # bus_real_power_df = pd.DataFrame(index=grid.P.index[:tm], columns=dss.Circuit.AllBusNames(), data=bus_real_power)
    line_power_df = pd.DataFrame(index=df_index, columns=dss.Lines.AllNames(), data=abs(line_power))

    # Final point for the calculation of execution time
    end = time.time()

    print('\n|||------------------------- Simulation summary -------------------------|||\n')
    print('Simulation period = ', tm, 's')
    print('Elapsed time =', end - start, ' s')
    print('Network losses =', Energy_Loss, ' kWh')
    # print('Maximum line loading =', np.amax((np.amax(Ca_per))), ' %')
    if np.amax(P_f) > 0:
        print('Maximum direct power flow (from utility to end-users) =', 100 * np.amax(P_f) / trf_nom_power, ' %')
        print('Minimum direct power flow (from utility to end-users) =',
              100 * float(min([n for n in P_f if n > 0])) / trf_nom_power, ' %')
    if np.amax(-P_f) > 0:
        print('Maximum reverse power flow =', 100 * np.amax(-P_f) / trf_nom_power, ' %')
        print('Minimum reverse power flow =', 100 * float(min([n for n in -P_f if n > 0])) / trf_nom_power, ' %')
    else:
        print('No reverse power flow occurred')

    # user_input = input('Do you want to save the results in a .csv file? [Y/N] \n')
    # if (user_input == 'Y') or (user_input == 'y'):

    if save_results:
        if folder is None:
            cwd = os.getcwd()
            folder = os.path.join(cwd, grid.name + ' from {} to {}'.format(grid.P.index[starting_second], grid.P.index[ending_second-1]).replace(':', '_'))

        write_flag = False

        if os.path.exists(folder):
            user_input = input('Results already exist. Do you want to rewrite? [Y/N] \n')
            if user_input.lower() == 'y':
                write_flag = True
        else:
            os.makedirs(folder)
            write_flag = True

        if write_flag:
            Va.to_csv(os.path.join(folder, 'Va.csv'))
            Vb.to_csv(os.path.join(folder, 'Vb.csv'))
            Vc.to_csv(os.path.join(folder, 'Vc.csv'))

            Ca.to_csv(os.path.join(folder, 'Ca.csv'))
            Cb.to_csv(os.path.join(folder, 'Cb.csv'))
            Cc.to_csv(os.path.join(folder, 'Cc.csv'))
            Cn.to_csv(os.path.join(folder, 'Cn.csv'))

            # Ca_per.to_csv(os.path.join(folder, 'Ca_per.csv'))
            # Cb_per.to_csv(os.path.join(folder, 'Cb_per.csv'))
            # Cc_per.to_csv(os.path.join(folder, 'Cc_per.csv'))
            # Cn_per.to_csv(os.path.join(folder, 'Cn_per.csv'))

            Losses.to_csv(os.path.join(folder, 'Losses.csv'))
            transformer_power.to_csv(os.path.join(folder, 'transformer_power.csv'))

            # bus_real_power_df.to_csv(os.path.join(folder, 'bus_real_power.csv'))
            line_power_df.to_csv(os.path.join(folder, 'line_power.csv'))

            # S substation
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

            # line length and c
            line_lengths = {}
            line_c = {}
            total_length = 0
            for k, line_ in enumerate(dss.Lines.AllNames()):
                dss.Lines.Idx(k + 1)
                line_lengths[line_] = dss.Lines.Length()
                total_length += line_lengths[line_]

                C = 1
                line_c[line_] = C

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

                if bus in load_names:
                    assert len(connections) == 1, 'House {} should be connected to a single bus.'.format(bus)
                else:
                    bus_connections[bus] = connections

            param_dict = {'folder': folder,
                          'S_substation': S_substation,
                          'initial_bus': initial_bus,
                          'line_length': line_lengths,
                          'total_length': total_length,
                          'line_c': line_c,
                          'bus_names': bus_names,
                          'bus_connections': bus_connections}

            with open('parameters.json', 'w') as fp:
                json.dump(param_dict, fp, sort_keys=True, indent=4)

            return os.path.join(os.getcwd(), 'parameters.json'), folder


    else:
        Va_vec = pd.DataFrame(index=df_index, data=np.array(Va_vec), columns=dss.Circuit.AllBusNames())
        Vb_vec = pd.DataFrame(index=df_index, data=np.array(Vb_vec), columns=dss.Circuit.AllBusNames())
        Vc_vec = pd.DataFrame(index=df_index, data=np.array(Vc_vec), columns=dss.Circuit.AllBusNames())

        return Va_vec, Vb_vec, Vc_vec, Ca, Cb, Cc, Cn
    #
    # else:
    #
    #     # SRC
    #     S_substation = 0
    #     for k, transformer in enumerate(dss.Transformers.AllNames()):
    #         dss.Transformers.Idx(k+1)
    #         temp = dss.Transformers.kVA()
    #         S_substation += temp
    #
    #     substation_reserve_capacity = pd.DataFrame(index=transformer_power.index, data=1-abs(S_trf_sec)/S_substation)
    #
    #     # FLLR
    #     # feeder_loss_to_load_ratio = pd.DataFrame(index=bus_real_power_df.index, columns=bus_real_power_df.columns,
    #     #                                          data=np.divide(Losses.P.values.reshape(tm, -1),
    #     #                                                         abs(1000*bus_real_power_df.values)))
    #
    #     total_load = grid[0].total_grid_power
    #     for i in range(1, len(grid.homes)):
    #         total_load += grid[i].total_grid_power
    #     for generator in grid.generators.values():
    #         total_load += generator.P.sum(axis=1)
    #
    #     feeder_loss_to_load_ratio = Losses.P/(abs(total_load))
    #
    #
    #     # ### PBI move to smart_grid ###
    #     bus_names = [bus for bus in dss.Circuit.AllBusNames()]
    #
    #     recursive_power_balance_index = {bus: None for bus in bus_names}
    #
    #     bus_connections = {}
    #     for bus in bus_names:
    #         connections = []
    #         for k, _ in enumerate(dss.Lines.AllNames()):
    #             dss.Lines.Idx(k + 1)
    #             bus1 = dss.Lines.Bus1().split('.')[0]
    #             bus2 = dss.Lines.Bus2().split('.')[0]
    #             if bus == bus1:
    #                 connections.append(bus2)
    #             elif bus == bus2:
    #                 connections.append(bus1)
    #
    #         if bus in load_names:
    #             assert len(connections) == 1, 'House {} should be connected to a single bus.'.format(bus)
    #             recursive_power_balance_index[bus] = grid.homes[bus].total_grid_power
    #         else:
    #             bus_connections[bus] = connections
    #
    #     def compute_pbi(bus, connections):
    #         pbi = 0
    #
    #         for connection in connections:
    #             if recursive_power_balance_index[connection] is None:
    #                 needed_buses = set(bus_connections[connection]) - {bus}
    #                 temp = compute_pbi(connection, list(needed_buses))
    #
    #                 if connection in list(grid.generators.keys()):
    #                     temp += grid.generators[connection].P.sum(axis=1)
    #
    #                 recursive_power_balance_index[connection] = temp
    #
    #             pbi += recursive_power_balance_index[connection]
    #
    #         return pbi
    #
    #     dss.Transformers.Idx(1)
    #     transformer_1_buses = [b.split('.')[0] for b in dss.CktElement.BusNames()]
    #     assert len(transformer_1_buses) == 2, 'There should be 2 substation buses'
    #     initial_bus = transformer_1_buses[1]
    #
    #     for k, transformer in enumerate(dss.Transformers.AllNames()):
    #         dss.Transformers.Idx(k+1)
    #         transformer_buses = [b.split('.')[0] for b in dss.CktElement.BusNames()]
    #         assert len(transformer_buses) == 2, 'There should be 2 substation buses'
    #         assert initial_bus == transformer_buses[1]
    #
    #     recursive_power_balance_index[initial_bus] = compute_pbi(initial_bus, bus_connections[initial_bus])
    #
    #
    #
    #     # afli
    #     afli = pd.DataFrame(index=line_power_df.index, columns=dss.Lines.AllNames(), data=0)
    #     total_length = 0
    #     for k, line_ in enumerate(dss.Lines.AllNames()):
    #         dss.Lines.Idx(k+1)
    #         s = abs(line_power[:, k])
    #         line_length = dss.Lines.Length()
    #         total_length += line_length
    #
    #         C = 1
    #
    #         afli[line_] = line_length*s/C
    #
    #     afli = afli/total_length
    #     afli = afli.sum(axis=1)
    #
    #     # new pbi
    #     power_balance_index = bus_real_power_df
    #
    #
    #     index_dict = {'SRC': substation_reserve_capacity, 'FFLR': feeder_loss_to_load_ratio,
    #                   'PBI': recursive_power_balance_index, 'AFLI': afli}
    #
    #     return index_dict


