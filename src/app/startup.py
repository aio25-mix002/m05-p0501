# import src.modeling.utils.randomness_utils as RandomnessUtils
# from src.configuration.configuration_manager import ConfigurationManager

GLOBAL_SEED = 42
TEST_SIZE = 0.25
def configure():
    # Load settings

    SETTINGS = {
        'random_state': 42,
        'test_size':0.25
    }

    return SETTINGS