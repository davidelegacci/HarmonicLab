import math as mt
import itertools
import utils
import numpy as np
import numpy.linalg as npla
from itertools import combinations
# from sympy import Matrix
import matplotlib.pyplot as plt

import common_methods as cm



class Player():
    """
    Player class. Each player has
    - name i
    - number of strategies Ai
    - strategies [1, 2, ..., Ai]
    """
    def __init__(self, num_strategies, player_name):
        self.num_strategies = num_strategies
        self.strategies = [i for i in range(1, num_strategies+1)]
        self.player_name = player_name

class Payoff():
    """game is Game instance"""
    def __init__(self, game, payoff_vector, plot_auction_potentialness = False, value = 0, pot_file_path = 0):
        """Either pass payoff vector as list, or generate random
        Integer payoff True by default; set false payoff can be float"""

        self.game = game




        self.payoff_vector = payoff_vector

        self.uN, self.uP, self.uH , self.potential = self.decompose_payoff()

        # equivalent game removing payoff component
        # self.u_strategic = self.uP + self.uH

        self.potentialness = self.measure_potentialness()

        if plot_auction_potentialness:
            # Value of 1-parameter plot_auction_potentialness game with 2 players for SLMath research
            self.value = value
            self.pot_file_path = pot_file_path
            cm.write_value_potentialness_FPSB(self)


        self.verbose_payoff()
        

    #################################################################################################
    # BEGIN PAYOFF METHODS
    #################################################################################################

    def decompose_payoff(self):

        # print('start decomposition')

        u = self.payoff_vector

        PI = self.game.normalization_projection
        e = self.game.exact_projection

        PWC_pinv = self.game.pwc_matrix_pinv
        PWC = self.game.pwc_matrix
        delta_0_pinv = self.game.coboundary_0_matrix_pinv
        delta_0 = self.game.coboundary_0_matrix

        # print('this seems to be the bottleneck, big matrices multiplication')

        uN = u - PI @ u
        # print('first multiplication done')

        uP = PWC_pinv @ e @ PWC @ u
        # print('three more multiplications done, this is slowest step!')

        uH = u - uN - uP

        phi = delta_0_pinv @ PWC @ u
        # print('two more multiplications done, end.')

        return [uN, uP, uH, phi]

    def round_list(self,L):
        # return [round(x,4) for x in L]
        return list(L)

    def measure_potentialness(self):
        '''NB this measures the size of the decomposed components naively using the Euclidean metric regardless of the metrics actually used in the decomposition.
        The metrics used in the decomposition are metrics on C0 and on C1, while the decomposed payoff lives in C0N that is a Cartesian product of copies of C0, but it's not
        obvious how to induce a metric on this space https://mathworld.wolfram.com/ProductMetric.html

        Measure "potentialness" as norm(u_p) / [ norm(u_p) + norm(u_h) ]
        If this number is 0 the game is purely harmonic
        If this number is 1 the game is purely potential
        '''
        uP_norm = float(((self.uP @ self.uP.T)[0][0]))**0.5
        uH_norm = float(((self.uH @ self.uH.T)[0][0]))**0.5
        potentialness = uP_norm / (uP_norm + uH_norm)
        return potentialness


    # def write_value_potentialness_FPSB(self):

    #     data = [[self.potentialness, self.value, self.game.num_strategies_for_player]]
    #     df = pd.DataFrame(data)
    #     df.to_csv(self.pot_file_path, header = False, index = False, mode='a', sep = ';')


    def verbose_payoff(self):
        print('\n-------------------- DECOMPOSITION  -----------------------')
        print(f'u = {self.payoff_vector}')
        print()
        print(f'uN = {self.round_list(self.uN)}')
        print(f'uP = {self.round_list(self.uP)}')
        print(f'uH = {self.round_list(self.uH)}')
        print()
        print(f'potential of uP = {self.round_list(self.potential)}')
        print('\n-------------------- SLMATH WORK ON DECOMPOSITION  -----------------------')
        print(f'Potentialness = {self.potentialness}')
        print('--------------------------------------\n')


class Game():
    def __init__(self, num_strategies_for_player):

        self.num_strategies_for_player = num_strategies_for_player
        self.num_players = len(num_strategies_for_player)

        # List of Player instances
        self.players = []
        for i in range(len(num_strategies_for_player)):
            self.players.append(Player(num_strategies_for_player[i], i+1))

        # A, that is number of strategis profiles, that is number of nodes of the response graph
        self.num_strategy_profiles = int(mt.prod(num_strategies_for_player))

        # AN, that is dimension of payoff space
        self.num_payoffs = self.num_strategy_profiles * self.num_players

        # Curly_A set, that is set of strategies profiles, of cardinality A
        # e.g. for 2x2 game it looks like [(1,1), (1,2), (2,1), (2,2)]
        self.strategy_profiles = list(itertools.product(*[p.strategies for p in self.players]))

        # Basis of (C^0)^N of cardinality AN, i.e. basis of vector space of payoffs
        # Its elements are e = (i,a) for i in N for a in A
        # e.g. for 2x2 it looks like [[0, (1, 1)], [0, (1, 2)], [0, (2, 1)], [0, (2, 2)], [1, (1, 1)], [1, (1, 2)], [1, (2, 1)], [1, (2, 2)]]
        self.payoff_basis = [ (i.player_name-1, a) for i in self.players for a in self.strategy_profiles ]

        ######################################################################
        # RESPONSE GRAPH
        ######################################################################

        # Count elements of response graph
        self.num_nodes = self.num_strategy_profiles
        self.num_edges_per_node = int(sum(self.num_strategies_for_player) - self.num_players)

        self.num_edges = int(self.num_nodes * self.num_edges_per_node / 2)
        # self.num_three_cliques = int(self.num_nodes / 6 * sum( [(ai-1)*(ai-2) for ai in self.num_strategies_for_player ] ))

        # Make Response Graph 
        self.nodes = self.strategy_profiles     # list [a, b, c]
        self.graph = self.make_graph()          # dictionary graph[a] = [b, c, d]
        self.edges = self.make_edges()          # list [ [a,b], [c,d] ]
        self.sort_elementary_chains(self.edges)

        # Simplicial complexes terminology
        self.dim_C0 = self.num_strategy_profiles
        self.dim_C1 = self.num_edges
        self.dim_C0N = self.num_payoffs
    
        ######################################################################
        # PWC MATRIX C0N --> C1
        self.pwc_matrix = self.make_pwc_matrix()
        ######################################################################


        ######################################################################
        # MATRIX coboundary 0 map
        # d_0: C^0 --> C^1
        self.coboundary_0_matrix = self.make_coboundary_0_matrix()
        ######################################################################

        ######################################################################
        # Pseudo-Inverse and projection block
        # Moore-Penrose pseudo-inverse of pwc
        # print('start PINV block')
        self.pwc_matrix_pinv = npla.pinv(self.pwc_matrix)

        # PI: C0N --> C0N projection onto Euclidean orthogonal complement of ker(δ_0^N)
        self.normalization_projection = np.matmul(self.pwc_matrix_pinv, self.pwc_matrix)

        # pinv(δ_0): C^1 --> C^0
        self.coboundary_0_matrix_pinv = npla.pinv(self.coboundary_0_matrix)

        # e: C1 --> C1 projection onto exact
        self.exact_projection = np.matmul(self.coboundary_0_matrix, self.coboundary_0_matrix_pinv)

        # Make potential of potential component
        # C0N --> C0
        # This is the matrix of the potential, i.e. map: utility --> its potential function
        # with utility in C0N and potential function in C0
        # The potential function itself is a function: A --> R
        # Ordered as basis of C0, e.g. in 2x3 case as [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
        # self.potential = np.matmul(self.coboundary_0_matrix_pinv, self.pwc_matrix)
        # print('end PINV block')

        self.verbose_game()
    #################################################################################################
    # BEGIN METHODS
    #################################################################################################

    # Make response graph dictionary, unoriented
    def make_graph(self):
        """
        Format: dictionary such that each key is a node and each value is the list of connected nodes
        e.g. graph[a] = [b, c, d] where a, b, c, d are nodes and [ab], [ac], [ad] are edges
        """
        graph = {}

        for s1 in self.strategy_profiles:
            unilateral_displacements = [s2 for s2 in self.strategy_profiles if utils.two_tuples_differ_for_one_element(s1,s2)]
            graph[s1] = unilateral_displacements

        return graph

    # Extract edges from responde graph
    def make_edges(self):
        """
        Format: list of lists, e.g. [ [a,b], [c,d] ] with [a,b] and [c,d] edges
        """
        edges = [ [A,B] for A in self.graph for B in self.graph[A] ]
        for e in edges:
            for f in edges:
                if e != f:
                    if utils.are_same_edge(e,f):
                        edges.remove(f)
        assert len(edges) == self.num_edges
        return edges

    #################################################################################################

    def make_pwc_matrix(self):
        """Matrix of pwc: C^O^N --> C^1"""
        A = np.zeros([int(self.dim_C1), int(self.dim_C0N)])

        for row in range(int(self.dim_C1)):
            edge = self.edges[row]
            i = utils.different_index(edge)

            minus_column = self.payoff_basis.index( (i, edge[0]) )
            plus_column = self.payoff_basis.index( (i, edge[1]) )
            A[row][minus_column] = -1
            A[row][plus_column] = +1

        return np.asmatrix(A)

    def make_coboundary_0_matrix(self):
        """Matrix of d_0: C^0 --> C^1"""

        # Start with transpose
        A = np.zeros([int(self.dim_C1), int(self.dim_C0)])

        for row in range(int(self.dim_C1)):
            basis_edge = self.edges[row]

            minus_node, plus_node = basis_edge
            minus_column = self.nodes.index( minus_node )
            plus_column = self.nodes.index( plus_node )
            A[row][minus_column] = -1
            A[row][plus_column] = +1

        return A  

    def sort_elementary_chains(self,list_of_simplices):
        for simplex in list_of_simplices:
            simplex.sort()

    def verbose_game(self):
        print(f'\nGAME: {self.num_strategies_for_player}')

