import pandas as pd

class PayoffGlobalMethods():
    def __init__(self, Payoff):
        '''Payoff is instance of Payoff class, either from fast or full code'''
        self.Payoff = Payoff

    def write_value_potentialness_FPSB(Payoff):
        data = [[Payoff.potentialness, Payoff.value, Payoff.game.num_strategies_for_player]]
        df = pd.DataFrame(data)
        df.to_csv(Payoff.pot_file_path, header = False, index = False, mode='a', sep = ';')


