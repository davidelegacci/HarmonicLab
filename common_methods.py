import pandas as pd

def find_pure_NE(Payoff):
    '''Payoff is Payoff class instance'''
    pure_NE = []
    receiving_nodes = [edge[-1] for edge in list(Payoff.unilateral_deviations_dict.keys())]
    for a in Payoff.game.nodes:
        if receiving_nodes.count(a) == Payoff.game.num_edges_per_node:
            pure_NE.append(a)
    return pure_NE


def write_value_potentialness_FPSB(Payoff):
    data = [[Payoff.potentialness, Payoff.value, Payoff.game.num_strategies_for_player]]
    df = pd.DataFrame(data)
    df.to_csv(Payoff.pot_file_path, header = False, index = False, mode='a', sep = ';')