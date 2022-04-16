import pandas as pd

from utils import pv_profile_generator


class PV:
    """
    Class representing solar panel.
    """

    def __init__(self, bus, pv_rated, single_phase, selected_phase='a', profile=0):
        """
        Initialization of the PV based on a dataframe as explained in the manual.
        :param bus: str
        :param pv_rated: float|int
        :param single_phase: bool
        :param selected_phase: str
        :param profile: int
        """

        self.bus = bus
        self.rated = pv_rated
        self.single_phase = single_phase
        self.selected_phase = selected_phase
        self.profile = profile
        self.P = pd.DataFrame()

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
            pv_values = pv_profile_generator(month, profile=self.profile)

        # initialize the total production series with 0s
        power = pd.Series(data=0,
                          name=days[0] + '-' + days[-1],
                          index=pd.timedelta_range(start='00:00:00', freq='1s', periods=len(days) * 60 * 60 * 24))

        # Calculate production for each day
        for k, day in enumerate(days):
            start_index = str(k) + ' days'
            end_index = start_index + ' 23:59:59'
            power[start_index:end_index] = self.rated * pv_values

        power *= 1000

        # initialize the production dataframe (per phase) with 0s
        three_phase_p = pd.DataFrame(data=0,
                                     columns=['a', 'b', 'c'],
                                     index=pd.timedelta_range(start='00:00:00', freq='1s',
                                                              periods=len(days) * 60 * 60 * 24))

        # if the panel is single-phase assign the total production to the correct phase
        if self.single_phase:
            three_phase_p[self.selected_phase] = power.values

        # if the panel is three-phase split between the phases
        else:
            for phase in three_phase_p.columns:
                three_phase_p[phase] = power.values/3

        self.P = three_phase_p

