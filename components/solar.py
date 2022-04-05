
class PV:
    """
    Class representing solar panel at free node of the grid.
    """

    def __init__(self, bus, pv_info):
        """
        Initialization of the PV based on a dataframe as explained in the manual.

        :param bus: str, Name of the bus that the panel is connected to.
        :param pv_info: pd.DataFrame
        """
        self.bus = bus
        self.pv_rated = pv_info['rated']
        self.P = None
        self.single_phase = True if pv_info['phase_number'] in [1, 2, 3] else False

        self.pv_profile = 0 if ('PV_profile' not in pv_info.index or np.isnan(pv_info['PV_profile'])) \
            else pv_info['PV_profile']

        if self.single_phase:
            assert pv_info['phase_number'] in [1, 2, 3]
            self.selected_phase = PHASES[pv_info['phase_number']-1]
        else:
            self.selected_phase = None

    def set_power(self, days, month, from_array=False, pv_array=None):
        """
        This method creates the solar production based on the month.
        :param days: [str]
        :param month: int
        :param from_array: bool
        :param pv_array: np.array
        :return:
        """
        if from_array:
            pv_values = pv_array
        else:
            pv_values = pv_profile_generator(month, profile=self.pv_profile)

        # initialize the total production series with 0s
        self.P = pd.Series(0, name=days[0] + '-' + days[-1],
                           index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days) * 60 * 60 * 24))

        # Calculate production for each day
        for k, day in enumerate(days):
            start_index = str(k) + ' days'
            end_index = start_index + ' 23:59:59'
            self.P[start_index:end_index] = self.pv_rated * pv_values

        self.P *= 1000

        # initialize the production dataframe (per phase) with 0s
        three_phase_p = pd.DataFrame(0, columns=['a', 'b', 'c'],
                                     index=pd.timedelta_range(start='00:00:00', freq='1s',
                                                              periods=len(days) * 60 * 60 * 24))

        # if the panel is single-phase assign the total production to the correct phase
        if self.single_phase:
            three_phase_p[self.selected_phase] = self.P.values

        # if the panel is three-phase split between the phases
        else:
            for phase in three_phase_p.columns:
                three_phase_p[phase] = self.P.values/3

        self.P = three_phase_p


