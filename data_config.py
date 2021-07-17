import os
import pathlib

ABSOLUTE_PATH = pathlib.Path(__file__).parent.absolute()

PROFILES_PATH = os.path.join(ABSOLUTE_PATH, 'dataset', 'profiles')

APPLIANCES = set(os.listdir(PROFILES_PATH)) - {'PV', 'AlwaysOn'}

NUM_APPLIANCES = len(APPLIANCES)

