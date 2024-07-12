import math as mt
import itertools
import utils
import numpy as np
import numpy.linalg as npla
from itertools import combinations
import matplotlib.pyplot as plt

from itertools import combinations
import networkx as nx
import solve_linear_system as sl
import sympy as sp
from sympy import Matrix
import pprint
import metric
import nashpy as nash
import pandas as pd

DIAGONAL_SHIFT_0 = 1
DIAGONAL_SHIFT_1 = 1
# kwargs[latex_plot_potential] = False

# MANUAL_METRICS

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
    """Basic"""
    def __init__(self, game, payoff_vector, **kwargs):
        """Either pass payoff vector as list, or generate random
        Integer payoff True by default; set false payoff can be float"""
        print('\nInit Payoff basic class...')
        self.game = game

        self.kwargs = kwargs

        self.payoff_vector = payoff_vector

        self.uN, self.uP, self.uH , self.potential = self.decompose_payoff()
        self.Du, self.DuP, self.DuH = self.hodge_decomposition()

        try:
            self.potentialness, self.potentialness_new = self.measure_potentialness()
            self.flow_potentialness, self.flow_potentialness_new = self.measure_flow_potentialness()
        except:
            print('Non strategic, cannot measure potentialness')
            self.potentialness, self.potentialness_new = [-1, -1]
            self.flow_potentialness, self.flow_potentialness_new = [-1, -1]


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

    def hodge_decomposition(self):
        '''Perform decomposition in flows space, forget about pseudo-inverse'''

        # print('start decomposition')

        u = self.payoff_vector

        e = self.game.exact_projection

        PWC = self.game.pwc_matrix

        # V1
        # Du = PWC @ u
        # DuP = e @ Du.T
        # DuH = Du - DuP.T

        # V2
        Du = PWC @ u
        DuP = e @ PWC @ u
        DuH = Du - DuP

        return [Du, DuP, DuH]

    def round_matrix(self,L):
        L =  (L.flatten()).tolist()[0]
        return [round(x,4) for x in L]

    def round_list(self,L):
        try:
            return [round(x,4) for x in L]
        except:
            L =  (L.flatten()).tolist()[0]
            return [round(x,4) for x in L]


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

        uPH = self.uP + self.uH
        uPH_norm = float(((uPH @ uPH.T)[0][0]))**0.5
        potentialness_new = uP_norm / uPH_norm

        return potentialness, potentialness_new



    def measure_flow_potentialness(self):
        '''
        Measure Euclidean norm of FLOWS in C1, so that non-strategic component does not play any role
        '''
        DuP_norm = float(((self.DuP @ self.DuP.T)[0][0]))**0.5
        DuH_norm = float(((self.DuH @ self.DuH.T)[0][0]))**0.5
        flow_potentialness = DuP_norm / (DuP_norm + DuH_norm)

        DuPH = self.DuP + self.DuH
        DuPH_nowm = float(((DuPH @ DuPH.T)[0][0]))**0.5
        flow_potentialness_new = DuP_norm / DuPH_nowm

        return flow_potentialness, flow_potentialness_new


    def write_value_potentialness_FPSB(self):
        data = [[self.potentialness, self.running_value, self.game.num_strategies_for_player]]
        df = pd.DataFrame(data)
        df.to_csv(self.kwargs['potentialness_file'], header = False, index = False, mode='a', sep = ';')


    def verbose_payoff(self):
        print(utils.red('\n-------------------- PAYOFF VERBOSE  -----------------------'))
        print('\n-------------------- DECOMPOSITION  -----------------------')
        print(f'u = {self.round_list(list(self.payoff_vector))}')
        print()
        print(f'uN = {self.round_matrix(self.uN)}')
        print(f'uP = {self.round_matrix(self.uP)}')
        print(f'uH = {self.round_matrix(self.uH)}')
        print()
        if self.game.num_players == 2: # and self.game.num_strategies_for_player[0] == self.game.num_strategies_for_player[1]:
            print('\n-------------------- DECOMPOSITION - matrix form for 2 players case-----------------------')
            print(f'u = {self.round_list(list(self.payoff_vector))}')
            print()
            A1, A2 = utils.uu_to_A( self.payoff_vector, self.game.num_strategies_for_player )
            AN1, AN2 = utils.uu_to_A( self.round_matrix(self.uN), self.game.num_strategies_for_player )
            AP1, AP2 = utils.uu_to_A( self.round_matrix(self.uP), self.game.num_strategies_for_player )
            AH1, AH2 = utils.uu_to_A( self.round_matrix(self.uH), self.game.num_strategies_for_player )
            print('u')
            print(A1)
            print()
            print(A2)
            print()
            print('uN')
            print(AN1)
            print()
            print(AN2)
            print()
            print('uP')
            print(AP1)
            print()
            print(AP2)
            print()
            print('uH')
            print(AH1)
            print()
            print(AH2)
            print()
        print(f'potential of uP = {self.round_matrix(self.potential)}')
        print('\n-------------------- SLMATH WORK ON DECOMPOSITION  -----------------------')
        print(f'Potentialness = {self.potentialness}')
        print(f'Potentialness new = {self.potentialness_new}')
        print('--------------------------------------\n')
        print('\n-------------------- FLOWS DECOMPOSITION  -----------------------')
        print()
        print(f'Du = {self.round_matrix(self.Du)}')
        print(f'DuP = {self.round_matrix(self.DuP)}')
        print(f'DuH = {self.round_matrix(self.DuH)}')
        print(f'\nFlow potentialness = {self.flow_potentialness}')
        print(f'Flow potentialness new = {self.flow_potentialness_new}')


class PayoffPotValue(Payoff):
    def __init__(self, game, payoff_vector, **kwargs):
        super().__init__(game, payoff_vector, **kwargs)
        print('\nInit Payoff Potentialness / Value class...')

        if kwargs['plot_auction_potentialness']:
            self.running_value = kwargs['running_value']
            self.write_value_potentialness_FPSB()

class PayoffNE(Payoff):
    def __init__(self, game, payoff_vector, **kwargs):
        super().__init__(game, payoff_vector, **kwargs)
        print('\nInit Payoff NE class...')

        self.payoff_dict = self.make_payoff_dict()

        self.bimatrix_form = self.make_bimatrix_form()
        self.NE = self.find_NE()

        print('start unil dev...')
        self.unilateral_deviations_dict = self.make_unilateral_deviations_dict()
        self.unilateral_deviations_dict = self.fix_edges_orientation()

        print('start pure NE...')
        self.pure_NE = self.find_pure_NE()

        self.verbose_payoff_NE()

    def make_payoff_dict(self):
        """
        The output is a dictionary assigning the payoff to each player and strategy profile, taking as input elements of game.payoff_basis 
        So u[e] = u_i(a) for e = (i,a) in self.payoff_basis
        It is true that self.payoff_dict[e] == self.payoff_function[a][i] for e = (i,a)
        """
        return dict(zip(self.game.payoff_basis, self.payoff_vector))

    def make_bimatrix_form(self):

        if self.game.num_players != 2:
            print(f'The payoff can be put in bimatrix form only for bimatrix games, but this game has {self.game.num_players} players')
            return

        rows, cols = self.game.num_strategies_for_player
        A, B = np.zeros((rows, cols)), np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                A[i][j] = self.payoff_dict[ (0, (i+1, j+1) ) ]
                B[i][j] = self.payoff_dict[ (1, (i+1, j+1) ) ]

        return A, B

    def find_NE(self):
        if self.game.num_players != 2:
            return
        A, B = self.bimatrix_form

        G = nash.Game(A,B)

        # Nash equilibria

        try:
            NE_1 = G.support_enumeration()
        except:
            print('Nash support enumeration failed')
        try:
            NE_2 = G.vertex_enumeration()
        except: print('Nash vertex enumeration failed')
        try:
            NE_3 = G.lemke_howson_enumeration()
        except: print('Nash support enumeration failed')

        return [NE_1, NE_2, NE_3]

    def find_pure_NE(self):
        '''self is self class instance'''
        pure_NE = []
        pure_NE_matrix = []
        zero_edges = [edge for edge in self.unilateral_deviations_dict.keys() if self.unilateral_deviations_dict[edge] == 0 ]
        receiving_nodes = [edge[-1] for edge in self.unilateral_deviations_dict.keys()] + [edge[0] for edge in zero_edges]
        for a in self.game.nodes:
            if receiving_nodes.count(a) == self.game.num_edges_per_node:
                pure_NE.append(a)
                pure_NE_matrix.append(1)
            else:
                pure_NE_matrix.append(0)

        if self.game.num_players == 2:
            return pure_NE, np.array(pure_NE_matrix).reshape(self.game.num_strategies_for_player)
        else:
            return pure_NE

    def print_NE(self):
        if self.game.num_players != 2:
            print(f'At the moment can find mixed NE only for bimatrix games')
            return
        NE_1, NE_2, NE_3 = self.NE
        print('\n--------------NASH--------------')
        print('\n------support_enumeration------')
        [print(ne) for ne in NE_1]
        print('\n------vertex_enumeration------')
        [print(ne) for ne in NE_2]
        print('\n------lemke_howson_enumeration------')
        [print(ne) for ne in NE_3]


    def make_unilateral_deviations_vector(self):
        """Returns vector of unilateral deviations, that is action of pwc matrix on payoff vector
        Its size is equal to dim C1"""

        v = np.matmul(self.game.pwc_matrix, self.payoff_vector).tolist()[0]
        assert(len(v) == self.game.dim_C1)
        return v

    def make_unilateral_deviations_dict(self):
        """Assigns edge to corresponding unilateral deviation, that is
        returns dictionary[edge] = actor's payoff difference.
        In other words, acts as the function in C^1 PWC(u), where u is payoff and PWC: C0N --> C1"""

        return dict(zip(self.game.networkx_graph.edges, self.make_unilateral_deviations_vector()))

    def fix_edges_orientation(self):
        """Re-orient edges such that all actor's deviations are non negative"""

        new_dict = {}

        for edge in self.unilateral_deviations_dict:
            value = self.unilateral_deviations_dict[edge]
            if value < 0:
                new_edge = ( edge[1], edge[0] )
                new_dict[new_edge] = - value
            else:
                new_dict[edge] = value

        return new_dict

    def verbose_payoff_NE(self):
        print(utils.red('\n-------------------- PAYOFF NE VERBOSE  -----------------------'))
        print('Bimatrix form\n')
        print(self.bimatrix_form)

        if self.game.num_players == 2:
            A, B = self.bimatrix_form
            print(sp.latex(sp.Matrix(A)))
            print(sp.latex(sp.Matrix(B)))

        print('\n-------------------- UNILATERAL DEVIATIONS  -----------------------')
        pprint.pprint(self.unilateral_deviations_dict)

        print('\n-------------------- NASH EQUILIBRIA  -----------------------')
        try:
            self.print_NE()
        except:
            print('Nash equilibria failed')
        print('\n------Pure NE------')
        print(self.pure_NE)

class PayoffFull(PayoffNE):

        def __init__(self, game, payoff_vector, **kwargs):
            """Basic + NE + Extra info"""
            super().__init__(game, payoff_vector, **kwargs)
            print('\nInit Payoff Full class...')

            self.payoff_vector_sympy = sp.Matrix(self.payoff_vector)
            self.kwargs = kwargs

            
            self.payoff_function = self.make_payoff_function()
            self.assert_payoff_consistency()
            


            self.verbose_payoff_full()


        def make_payoff_function(self):
            """
            The output is a dictionary working like the payoff function u: A --> R^N
            So u[a] = (u_1(a), ..., u_N(a)) for a in curly_A, that is for a in game.strategy_profiles
            It is true that self.payoff_dict[e] == self.payoff_function[a][i] for e = (i,a)
            """
            payoff_function = {}

            for strategy_profile in self.game.strategy_profiles:
                po = []
                for pl in self.game.players:
                    x = self.payoff_dict[(pl.player_name-1, strategy_profile)]
                    po.append(float(x))

                payoff_function[strategy_profile] = np.array(po)

            return payoff_function

        def assert_payoff_consistency(self):
            for e in self.game.payoff_basis:
                i, a = e
                assert self.payoff_dict[e] == self.payoff_function[a][i]

        # def payoff_difference(self, strategy_profile_1, strategy_profile_2):
        #     return self.payoff_function[strategy_profile_1] - self.payoff_function[strategy_profile_2]

        def is_harmonic(self):
            Z = self.game.harmonic_matrix_sympy * self.payoff_vector_sympy
            return utils.is_zero(Z)

        def is_euclidean_harmonic(self):
            Z = self.game.harmonic_matrix_sympy_euclidean * self.payoff_vector_sympy
            return utils.is_zero(Z)

        

        def draw_directed_response_graph(self):

            # directed graph
            G = nx.DiGraph()
            G.add_nodes_from(self.game.nodes)

            oriented_edges = self.unilateral_deviations_dict.keys()
            G.add_edges_from(oriented_edges)
            # G.add_edges_from(self.unilateral_deviations_dict)

            draw_options = {

            'with_labels' : True,
            'font_weight' : 'bold',
            'arrows' : True,
            # 'node_color': 'blue',
            # 'node_size': 100,
            # 'width': 3,
            # 'arrowstyle': '-|>',
            'arrowsize': 20,
            }

            pos = nx.spring_layout(G)
            nx.draw(G, pos, **draw_options)
            nx.draw_networkx_edge_labels(G, pos, edge_labels = self.unilateral_deviations_dict, font_color = 'red')

            plt.show()

        def make_latex_graph_2x2_code(self, potential = False):

            if self.game.num_strategies_for_player != [2,2]:
                return

            dev_dict = self.unilateral_deviations_dict
            edges = list( dev_dict.keys() )

            # kill trailing zeros https://stackoverflow.com/questions/2440692/formatting-floats-without-trailing-zeros
            def nice(x):
                return '%g'%(x)

            def get_deviation(edge):
                """edge = (a, b)
                Try to get deviation of edge from dictionary; if error, need to change edge orientation"""

                a, b = edge

                try:
                    dev = dev_dict[edge]
                except:
                    dev = dev_dict[ (b,a) ]

                return nice(dev)

            def get_payoff(node):
                a, b = self.payoff_function[node]
                return f'{nice(round(a,2))},{nice(round(b,2))}'

            # Adjust edges orientation in LaTex graph
            ar_a = '->' if ((1, 1), (1, 2)) in edges else '<-'
            ar_d = '->' if ((2, 1), (2, 2)) in edges else '<-'
            ar_g = '->' if ((1, 1), (2, 1)) in edges else '<-'
            ar_h = '->' if ((1, 2), (2, 2)) in edges else '<-'


            ######################################
            a =  get_deviation( ((1, 1), (1, 2)) )
            if float(a) < 1e-3: ar_a = '<->'
            
            
            d =  get_deviation( ((2, 1), (2, 2)) )
            if float(d) < 1e-3: ar_d = '<->'
            
            
            g =  get_deviation( ((1, 1), (2, 1)) )
            if float(g) < 1e-3: ar_g = '<->'
            
            h =  get_deviation( ((1, 2), (2, 2)) )
            if float(h) < 1e-3: ar_h = '<->'
            
            ###################################
            # manual_potential = [0, 1, 1, 2]


            ##################################
            # if potential include potential function at nodes
            if potential:
                nodes_latex_code = f"""
        \x5cnode[main, label={{left:\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{nice(round(self.potential[0],2))}]}}}}({get_payoff((1,1))}) }}] (1) {{$11$}};
        \x5cnode[main, label={{right:\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{nice(round(self.potential[1],2))}]}}}}({get_payoff((1,2))})}}] (2) [right of=1] {{$12$}};
        \x5cnode[main, label={{left:\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{nice(round(self.potential[2],2))}]}}}}({get_payoff((2,1))})}}] (3) [above of=1] {{$21$}};
        \x5cnode[main, label={{right:\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{nice(round(self.potential[3],2))}]}}}}({get_payoff((2,2))})}}] (4) [above of=2] {{$22$}}; 

                """


            if not potential:
                nodes_latex_code = f"""
        \x5cnode[main, label={{left:\x5csmall {get_payoff((1,1))}}}] (1) {{$11$}};
        \x5cnode[main, label={{right:\x5csmall {get_payoff((1,2))}}}] (2) [right of=1] {{$12$}};
        \x5cnode[main, label={{left:\x5csmall {get_payoff((2,1))}}}] (3) [above of=1] {{$21$}};
        \x5cnode[main, label={{right:\x5csmall {get_payoff((2,2))}}}] (4) [above of=2] {{$22$}}; 
                """
            ##################################

            LaTex_code = f"""\x5cbegin{{tikzpicture}}[node distance={{17mm}}, thick, main/.style = {{draw, circle}}]

            {nodes_latex_code}

        % Horizontal down
        \x5cdraw [{ar_a}, red] (1) -- node[below]{{${a}$}} (2);

        % Horizontal up
        \x5cdraw [{ar_d}, red] (3) -- node[above]{{${d}$}} (4);

        % Vertical left
        \x5cdraw [{ar_g}, blue] (1) -- node[left]{{${g}$}} (3);

        % Vertical right
        \x5cdraw [{ar_h}, blue] (2) -- node[left]{{${h}$}} (4);
        \x5cend{{tikzpicture}}"""

            print(LaTex_code)


        ####################################################################

        def make_latex_graph_2x3_code(self, potential = False):

            if self.game.num_strategies_for_player != [2,3]:
                return

            dev_dict = self.unilateral_deviations_dict
            edges = list( dev_dict.keys() )

            # kill trailing zeros https://stackoverflow.com/questions/2440692/formatting-floats-without-trailing-zeros
            def nice(x):
                return '%g'%(x)

            def get_deviation(edge):
                """edge = (a, b)
                Try to get deviation of edge from dictionary; if error, need to change edge orientation"""

                a, b = edge

                try:
                    dev = dev_dict[edge]
                except:
                    dev = dev_dict[ (b,a) ]

                return nice(dev)

            def get_payoff(node):
                a, b = self.payoff_function[node]
                return f'{nice(round(a,2))}, {nice(round(b,2))}'

            # Adjust edges orientation in LaTex graph
            ar_a = '->' if ((1, 1), (1, 2)) in edges else '<-'
            ar_b = '->' if ((1, 2), (1, 3)) in edges else '<-'
            ar_c = '->' if ((1, 1), (1, 3)) in edges else '<-'
            ar_d = '->' if ((2, 1), (2, 2)) in edges else '<-'
            ar_e = '->' if ((2, 2), (2, 3)) in edges else '<-'
            ar_f = '->' if ((2, 1), (2, 3)) in edges else '<-'
            ar_g = '->' if ((1, 1), (2, 1)) in edges else '<-'
            ar_h = '->' if ((1, 2), (2, 2)) in edges else '<-'
            ar_i = '->' if ((1, 3), (2, 3)) in edges else '<-'


            ######################################
            a =  get_deviation( ((1, 1), (1, 2)) )
            if float(a) < 1e-3: ar_a = '<->'
            
            b =  get_deviation( ((1, 2), (1, 3)) )
            if float(b) < 1e-3: ar_b = '<->'
            
            c =  get_deviation( ((1, 1), (1, 3)) )
            if float(c) < 1e-3: ar_c = '<->'
            
            d =  get_deviation( ((2, 1), (2, 2)) )
            if float(d) < 1e-3: ar_d = '<->'
            
            e =  get_deviation( ((2, 2), (2, 3)) )
            if float(e) < 1e-3: ar_e = '<->'
            
            f =  get_deviation( ((2, 1), (2, 3)) )
            if float(f) < 1e-3: ar_f = '<->'
            
            g =  get_deviation( ((1, 1), (2, 1)) )
            if float(g) < 1e-3: ar_g = '<->'
            
            h =  get_deviation( ((1, 2), (2, 2)) )
            if float(h) < 1e-3: ar_h = '<->'
            
            i =  get_deviation( ((1, 3), (2, 3)) )
            if float(i) < 1e-3: ar_i = '<->'
            ###################################


            ##################################
            # if potential include potential function at nodes
            if potential:
                nodes_latex_code = f"""
        \x5cnode[main, label={{[xshift=-1.5cm, yshift=-1cm]\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{round(self.potential[0],2)}]}}}}({get_payoff((1,1))}) }}] (1) {{$11$}};
        \x5cnode[main, label={{[xshift=-0.0cm, yshift=-1.4cm]\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{round(self.potential[1],2)}]}}}}({get_payoff((1,2))})}}] (2) [right of=1] {{$12$}};
        \x5cnode[main, label={{[xshift=+1.5cm, yshift=-1cm]\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{round(self.potential[2],2)}]}}}}({get_payoff((1,3))})}}] (3) [right of=2] {{$13$}};
        \x5cnode[main, label={{[xshift=-1.5cm, yshift=-1cm]\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{round(self.potential[3],2)}]}}}}({get_payoff((2,1))})}}] (4) [above of=1] {{$21$}};
        \x5cnode[main, label={{[xshift=-0.0cm, yshift=-0.05cm]\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{round(self.potential[4],2)}]}}}}({get_payoff((2,2))})}}] (5) [right of=4] {{$22$}}; 
        \x5cnode[main, label={{[xshift=+1.5cm, yshift=-1cm]\x5csmall \x5ctextbf{{\x5ctextcolor{{teal}}{{[{round(self.potential[5],2)}]}}}}({get_payoff((2,3))})}}] (6) [right of=5] {{$23$}};
                """

            if not potential:
                nodes_latex_code = f"""
        \x5cnode[main, label={{[xshift=-0.8cm, yshift=-1cm]\x5csmall {get_payoff((1,1))}}}] (1) {{$11$}};
        \x5cnode[main, label={{[xshift=-0.0cm, yshift=-1.4cm]\x5csmall {get_payoff((1,2))}}}] (2) [right of=1] {{$12$}};
        \x5cnode[main, label={{[xshift=+0.8cm, yshift=-1cm]\x5csmall {get_payoff((1,3))}}}] (3) [right of=2] {{$13$}};
        \x5cnode[main, label={{[xshift=-0.8cm, yshift=-1cm]\x5csmall {get_payoff((2,1))}}}] (4) [above of=1] {{$21$}};
        \x5cnode[main, label={{[xshift=-0.0cm, yshift=-0.05cm]\x5csmall {get_payoff((2,2))}}}] (5) [right of=4] {{$22$}}; 
        \x5cnode[main, label={{[xshift=+0.8cm, yshift=-1cm]\x5csmall {get_payoff((2,3))}}}] (6) [right of=5] {{$23$}};
                """
            ##################################

            LaTex_code = f"""\x5cbegin{{tikzpicture}}[node distance={{17mm}}, thick, main/.style = {{draw, circle}}]

            {nodes_latex_code}

        % Horizontal down
        \x5cdraw [{ar_a}, red] (1) -- node[below]{{${a}$}} (2);
        \x5cdraw [{ar_b}, red] (2) -- node[below]{{${b}$}} (3);

        % Curve down
        \x5cdraw [{ar_c}, red] (1) to [out = -90, in = -90 ] node[below]{{${c}$}} (3);

        % Horizontal up
        \x5cdraw [{ar_d}, red] (4) -- node[above]{{${d}$}} (5);
        \x5cdraw [{ar_e}, red] (5) -- node[above]{{${e}$}} (6);

        % Curve up
        \x5cdraw [{ar_f}, red] (4) to [out = 90, in = 90 ] node[above]{{${f}$}} (6);

        % Vertical
        \x5cdraw [{ar_g}, blue] (1) -- node[left]{{${g}$}} (4);
        \x5cdraw [{ar_h}, blue] (2) -- node[left]{{${h}$}} (5);
        \x5cdraw [{ar_i}, blue] (3) -- node[right]{{${i}$}} (6);
        \x5cend{{tikzpicture}}"""

            print(LaTex_code)


        ####################################################################


        def verbose_payoff_full(self):
            print('\n------------------------------------------------------------')
            print(utils.red('\n-------------------- PAYOFF FULL VERBOSE  -----------------------'))
            print('Payoff function')
            pprint.pprint(self.payoff_function)
            print('\nIndividual payoff')
            pprint.pprint(self.payoff_dict)
            print(utils.orange(f'\nMetric can be changed in config.yml. Current: {self.game.metric_type}'))
            print(utils.orange(f'\nIs harmonic (in N+H) wrt current metric: {self.is_harmonic()}'))
            print(utils.orange(f'Is harmonic (in N+H) wrt euclidean metric: {self.is_euclidean_harmonic()}'))
            print('\n-------------------- LATEX 2X3 GRAPH  -----------------------')
            self.make_latex_graph_2x3_code(potential = self.kwargs['latex_plot_potential'])
            self.make_latex_graph_2x2_code(potential = self.kwargs['latex_plot_potential'])


class Game():
    def __init__(self, num_strategies_for_player, **kwargs):

        print(utils.orange('\n-------------------------------------------------------------------------------------------'))
        print(utils.orange('\n-------------------- INIT BASIC GAME CLASS - EXECUTION STARTS HERE  -----------------------'))
        print(utils.orange('\n-------------------------------------------------------------------------------------------'))

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
        
        # original ordering
        self.payoff_basis = [ (i.player_name-1, a) for i in self.players for a in self.strategy_profiles ]

        # switch ordering
        #self.payoff_basis = [ (i.player_name-1, a) for a in self.strategy_profiles for i in self.players  ]

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
        print(utils.red('\n-------------------- GAME VERBOSE  -----------------------'))
        print(f'\nGAME: {self.num_strategies_for_player}')







        

class GameFull(Game):
    """
    Basic + NE
    """
    def __init__(self, num_strategies_for_player, **kwargs):
        super().__init__(num_strategies_for_player)
        print('\nInit Game full class...')

        """num_strategies_for_player is list, e.g. [3] for 1 player 3 strategies, [2,2] for 2 players 2 strategies each, etc.
        If input metric manually, input matrix A such that metric is g = A*At
        Two matrices are needed, one of dimension = dim C0 and one of dimension = dim C1
        In this case the manual_metrics input is a list of these two matrices [A, B]
        """

        self.kwargs = kwargs
        self.metric_type = kwargs['metric_type']
        assert(self.metric_type in ['euclidean', 'random', 'manual', 'diagonal', 'harmonic'])
        



        ######################################################################
        # GAME
        ######################################################################
        # print('\nInit game...')
        # self.num_strategies_for_player = num_strategies_for_player
        
        # Number of players
        # self.num_players = len(num_strategies_for_player)

        # List of Player instances
        # self.players = []
        # for i in range(len(num_strategies_for_player)):
        #     self.players.append(Player(num_strategies_for_player[i], i+1))

        # A, that is number of strategis profiles, that is number of nodes of the response graph
        # self.num_strategy_profiles = int(mt.prod(num_strategies_for_player))

        # AN, that is dimension of payoff space
        # self.num_payoffs = self.num_strategy_profiles * self.num_players

        # Curly_A set, that is set of strategies profiles, of cardinality A
        # e.g. for 2x2 game it looks like [(1,1), (1,2), (2,1), (2,2)]
        # self.strategy_profiles = list(itertools.product(*[p.strategies for p in self.players]))


        # Basis of (C^0)^N of cardinality AN, i.e. basis of vector space of payoffs
        # Its elements are e = (i,a) for i in N for a in A
        # e.g. for 2x2 it looks like [[0, (1, 1)], [0, (1, 2)], [0, (2, 1)], [0, (2, 2)], [1, (1, 1)], [1, (1, 2)], [1, (2, 1)], [1, (2, 2)]]
        # self.payoff_basis = [ [i.player_name-1, a] for i in self.players for a in self.strategy_profiles ]
        # self.payoff_basis = [ (i.player_name-1, a) for i in self.players for a in self.strategy_profiles ]
        self.payoff_strings = [ 'u'+str(e) for e in self.payoff_basis ]

        # for s in self.payoff_strings:
        #     assert(type(s) == str)

        self.payoff_variables = [sp.Symbol(s) for s in self.payoff_strings]


        self.test_variables = []
        for i in range(1, len(self.payoff_basis)+1):
            tmp_st = f'x{i}'
            globals()[tmp_st] = sp.Symbol(tmp_st)
            self.test_variables.append(globals()[tmp_st])

        if kwargs['payoff_variable_format'] ==  'x':
            self.payoff_variables = self.test_variables

        ######################################################################
        # RESPONSE GRAPH
        ######################################################################
        # print('Start response graph..')
        # Count elements of response graph
        # self.num_nodes = self.num_strategy_profiles
        # self.num_edges_per_node = int(sum(self.num_strategies_for_player) - self.num_players)

        # self.num_edges = int(self.num_nodes * self.num_edges_per_node / 2)
        self.num_three_cliques = int(self.num_nodes / 6 * sum( [(ai-1)*(ai-2) for ai in self.num_strategies_for_player ] ))

        # Make Response Graph 
        # self.nodes = self.strategy_profiles     # list [a, b, c]
        # self.graph = self.make_graph()          # dictionary graph[a] = [b, c, d]
        # self.edges = self.make_edges()          # list [ [a,b], [c,d] ]
        # self.sort_oriented_simplices(self.edges)

        # Make networkx Response Graph from graph dictionary
        # UNORIENTED
        self.networkx_graph = nx.Graph(self.graph)

        # Simplicial complexes terminology
        # self.dim_C0 = self.num_strategy_profiles
        # self.dim_C1 = self.num_edges
        self.dim_C2 = self.num_three_cliques
        # self.dim_C0N = self.num_payoffs

        ######################################################################
        # INNER PRODUCT
        # metric_0 and metric_1 are Metric instances for C0 and C1
        ######################################################################
        if self.metric_type == 'euclidean':
            self.metric_0 = metric.Metric( metric.EuclideanMetric(self.dim_C0).matrix )
            self.metric_1 = metric.Metric( metric.EuclideanMetric(self.dim_C1).matrix )

        elif self.metric_type == 'diagonal':
            self.metric_0 = metric.Metric( metric.DiagonalMetric(self.dim_C0, DIAGONAL_SHIFT_0).matrix )
            self.metric_1 = metric.Metric( metric.DiagonalMetric(self.dim_C1, DIAGONAL_SHIFT_1).matrix )

        elif self.metric_type == 'random':
            self.metric_0 = metric.Metric( metric.RandomMetric(self.dim_C0).matrix )
            self.metric_1 = metric.Metric( metric.RandomMetric(self.dim_C1).matrix )


        elif self.metric_type == 'manual':

            # specify here manual metrics

            g0 = np.eye( self.dim_C0 )
            g1 = np.eye( self.dim_C1 )

            # self.manual_metrics = [g0, g1]

            # Make sure square matrices
            assert( len(g0) == len(g0[0]))
            assert( len(g1) == len(g1[1]))

            # Make sure right dimension
            assert( len(g0) == self.dim_C0)
            assert( len(g1) == self.dim_C1)

            self.metric_0 = metric.Metric( g0 )
            self.metric_1 = metric.Metric( g1 )


        elif self.metric_type == 'harmonic':

            self.harmonic_measure = kwargs['harmonic_measure']
            assert len(self.harmonic_measure) == len(self.num_strategies_for_player), "harmonic measure has bad size"
            for i, m in enumerate(self.harmonic_measure):
                assert len(m) == self.num_strategies_for_player[i], "harmonic measure has bad size"

            g0 = np.eye( self.dim_C0 )
            g1 = np.eye( self.dim_C1 )

            # perturb C1 metric
            edges = self.networkx_graph.edges
            skeleton = self.num_strategies_for_player
            players = range(self.num_players)
            pures = self.nodes
            pures_play = [ p.strategies for p in self.players ]

            # HARMONIC MEASUIRE list of dicts, measure in C0N
            self.mu = [ dict(zip(pures_play[i], self.harmonic_measure[i])) for i in players  ]

            # PRODUCT MEASURE, measure on space of action profiles, C0
            # P stands for product; this is product measure
            muPvalues = [      np.prod(  [ self.mu[i][  a[i]  ]  for i in players   ]   )       for a in pures    ]
            self.muP = dict(zip(pures, muPvalues))

            # EDGES MEASURE, measure in space C1
            muEvalues = [    self.muP[ e[0] ] *  self.muP[ e[1] ]     for e in edges    ]
            self.muE = dict(zip(edges, muEvalues))

            #--------------------------------------------------------
            print(f"\nHarmonic measure: {self.harmonic_measure}, i.e.")

            for i, mui in enumerate(self.mu):
                print(f"\nMeasure player {i}")
                [ print( f" {ai} : {mui[ai]} " ) for ai in mui ]

            print("\nProduct measure on action profiles:")
            for a in pures:
                print(f"{a}: {self.muP[a]}")


            print("\nMeasure on edges:")
            for e in self.muE:
                print(f"{e}: {self.muE[e]}")
            #--------------------------------------------------------

            edges_measure = list(self.muE.values())
            assert len(edges_measure) == len(g1)

            # place edges measures on diagonal of C1 inner product (fraction and square root)
            for i, x in enumerate(edges_measure):

                g1[i][i] = 1 / ( np.sqrt(x) )

            # Make sure square matrices
            assert( len(g0) == len(g0[0]))
            assert( len(g1) == len(g1[1]))

            # Make sure right dimension
            assert( len(g0) == self.dim_C0)
            assert( len(g1) == self.dim_C1)

            self.metric_0 = metric.Metric( g0 )
            self.metric_1 = metric.Metric( g1 )
    
        ######################################################################
        # PWC MATRIX C0N --> C1
        ######################################################################
        # print('D matrix...')
        # Establish wether pwc is surjective from theory. Surjective iff all players have 1 or 2 strategies. 
        self.pwc_is_surjective = True
        for ai in self.num_strategies_for_player:
            if ai >= 3:
                self.pwc_is_surjective = False
                break

        # Make numpy pwc matrix
        # self.pwc_matrix = self.make_pwc_matrix()

        # Make sympy pwc matrix
        self.pwc_matrix_sympy = Matrix(self.pwc_matrix)

        # Kernel of pwc matrix = basis of N
        self.basis_nonstrategic_games = self.pwc_matrix_sympy.nullspace()

        self.dim_ker_pwc_theory = int(self.sum_prod(self.num_strategies_for_player))
        self.dim_im_pwc_theory = int(self.num_payoffs - self.dim_ker_pwc_theory)

        assert(self.dim_ker_pwc_theory == len(self.basis_nonstrategic_games))

        # Check whether pwc is surjective numerically for consistency. Surjective iff full rank. 
        self.dim_im_pwc_numerical = npla.matrix_rank(self.pwc_matrix)
        self.dim_ker_pwc_numerical = self.num_payoffs - self.dim_im_pwc_numerical
        self.pwc_is_surjective_numerical = True if self.dim_im_pwc_numerical == int(self.dim_C1) else False
        assert(self.dim_im_pwc_numerical == self.dim_im_pwc_theory)
        assert(self.dim_ker_pwc_numerical == self.dim_ker_pwc_theory)
        assert(self.pwc_is_surjective == self.pwc_is_surjective_numerical)


        ######################################################################
        # Hodge decomposition in C1
        ######################################################################
        # print('Cliques...')
        self.dim_exact = self.num_nodes - 1

        # Count three cliques iff there are three cliques, i.e. iff at least one player has 3 strategies or more iff pwc not surjective
        self.there_are_3_cliques = not self.pwc_is_surjective

        if self.there_are_3_cliques:
            # Get dim closed by computing explicitely the matrix of d_1 and finding the dimension of its kernel

            # get 3-cliques algorithmically
            self.three_cliques = self.make_cliques(self.networkx_graph, 3)

            # Sort basis elements; necessary to build matrix, so that basis elements match
            # self.sort_oriented_simplices(self.edges)
            self.sort_elementary_chains(self.three_cliques)

            # Count three-cliques
            self.algo_num_three_cliques = len(self.three_cliques)

            # Sometimes algorithm fails and includes duplicates; get rid of them
            if not self.num_three_cliques == self.algo_num_three_cliques:
                self.three_cliques = self.kill_duplicates(self.three_cliques)
                self.algo_num_three_cliques = len(self.three_cliques)
                assert self.algo_num_three_cliques == self.num_three_cliques            

            #####################################################################
            # Matrix of d_1: C^1 --> C^2
            #####################################################################
            self.coboundary_1_matrix = self.make_coboundary_1_matrix()
            self.coboundary_1_matrix_rank = npla.matrix_rank(self.coboundary_1_matrix)
            self.coboundary_1_matrix_kernel = self.num_edges - self.coboundary_1_matrix_rank
            self.dim_closed = self.coboundary_1_matrix_kernel


        if not self.there_are_3_cliques:
            # pwc is surjective, so dim closed is dim C1
            self.dim_closed = self.dim_C1

        # Dim harmonic = ANSATZ dim H
        # If pwc is surjectiye, dim harmonic is known in closed form as dim C1
        # If pwc is not surjective, dim harmononic is computed explicitely, ketting dim closed as dim ker d_1
        
        self.dim_harmonic = int(self.dim_closed - self.dim_exact)

        ######################################################################
        # Hodge decomposition in C0N
        ######################################################################
        self.dim_P = self.dim_exact
        self.dim_H = int(self.num_nodes * (self.num_players - 1 - sum( [ 1/ai for ai in self.num_strategies_for_player ] )) + 1)
        self.dim_N = self.dim_ker_pwc_theory

        # Here dim_H is known in closed form, since dim C0N, dimP and dimN are known in closed for
        assert(self.dim_harmonic == self.dim_H)

        ######################################################################
        # MATRIX boundary 1 map, dual of coboundary 0 map
        #(d_0)* : C_1 --> C_0
        # Dual of d_0: C^0 --> C^1
        self.boundary_1_matrix = self.coboundary_0_matrix.transpose()
        ######################################################################


        ######################################################################
        # Pseudo-Inverse block


        if self.metric_type == 'euclidean':

            # # Moore-Penrose pseudo-inverse of pwc
            # self.pwc_matrix_pinv = npla.pinv(self.pwc_matrix)

            # # PI: C0N --> C0N projection onto Euclidean orthogonal complement of ker(d_0^N)
            # self.normalization_projection = np.matmul(self.pwc_matrix_pinv, self.pwc_matrix)

            # # pinv(d_0): C^1 --> C^0
            # self.coboundary_0_matrix_pinv = npla.pinv(self.coboundary_0_matrix)

            # # e: C1 --> C1 projection onto exact
            # self.exact_projection = np.matmul(self.coboundary_0_matrix, self.coboundary_0_matrix_pinv)

            # Make potential of potential component
            # C0N --> C0
            # This is the matrix of the potential, i.e. map: utility --> its potential function
            # with utility in C0N and potential function in C0
            # The potential function itself is a function: A --> R
            # Ordered as basis of C0, e.g. in 2x3 case as [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
            self.potential_matrix = np.matmul(self.coboundary_0_matrix_pinv, self.pwc_matrix)
        ######################################################################
       
       
        ######################################################################
        # MATRIX adjoint of coboundary 0 map
        #(d_0)+ : C^1 --> C^0
        # Adjoint of d_0: C^0 --> C^1
        # Flat and sharp are wrt metrics in C_0 and C_1
        self.delta_0_adjoint_matrix = np.matmul(self.metric_0.flat_matrix, np.matmul(self.boundary_1_matrix, self.metric_1.sharp_matrix))
        ######################################################################

        ######################################################################
        # HARMONIC MATRIX
        ######################################################################
        # print('Multiply stuff..')

        # Harmonic matrix EUCLIDEAN = del_1 \circ pwc: C0N --> C1 --> C0

        self.harmonic_matrix = np.matmul(self.delta_0_adjoint_matrix, self.pwc_matrix)
        self.harmonic_matrix_sympy = Matrix(self.harmonic_matrix)

        # Identifying dual with adjoint
        # The harmonic payoffs N+H are the kernel of this matrix
        self.harmonic_matrix_euclidean = np.matmul(self.boundary_1_matrix, self.pwc_matrix)
        self.harmonic_matrix_sympy_euclidean = Matrix(self.harmonic_matrix_euclidean)

        self.dim_im_harmonic_matrix = npla.matrix_rank(self.harmonic_matrix)
        self.dim_ker_harmonic_matrix = self.dim_C0N - self.dim_im_harmonic_matrix
        assert(self.dim_ker_harmonic_matrix == self.dim_N + self.dim_H)

        # Study harmonic games
        # CAREFUL! The parametric solution given by sympy returns a finiteset object, and the order of the solutions seems to be messed up sometimes
        # Do not rely on parametric solution, use basis solutions
        # self.basis_harmonic_games containt sympy vectors

        self.harmonic_games_parametric, self.basis_harmonic_games = self.make_basis_harmonic_games(self.payoff_variables)

        try:
            assert(len(self.basis_harmonic_games) == self.dim_N + self.dim_H)
            self.harmonic_system_solved = True

        except:
            self.harmonic_system_solved = False

        self.random_harmonic_payoff = self.make_random_harmonic_payoff_int()

        self.verbose_game_full()

    #################################################################################################
    # BEGIN METHODS
    #################################################################################################


    # # Make response graph dictionary, unoriented
    # def make_graph(self):
    #     """
    #     Format: dictionary such that each key is a node and each value is the list of connected nodes
    #     e.g. graph[a] = [b, c, d] where a, b, c, d are nodes and [ab], [ac], [ad] are edges
    #     """
    #     graph = {}

    #     for s1 in self.strategy_profiles:
    #         unilateral_displacements = [s2 for s2 in self.strategy_profiles if utils.two_tuples_differ_for_one_element(s1,s2)]
    #         graph[s1] = unilateral_displacements

    #     return graph

    # # Extract edges from responde graph
    # def make_edges(self):
    #     """
    #     Format: list of lists, e.g. [ [a,b], [c,d] ] with [a,b] and [c,d] edges
    #     """
    #     edges = [ [A,B] for A in self.graph for B in self.graph[A] ]
    #     for e in edges:
    #         for f in edges:
    #             if e != f:
    #                 if utils.are_same_edge(e,f):
    #                     edges.remove(f)
    #     assert len(edges) == self.num_edges
    #     return edges

    #################################################################################################

    def slice(self, a, i):
        return a[:i]+a[i+1:]

    def sum_prod(self, a):
        return np.sum( [ np.prod(self.slice(a, i)) for i in range(len(a))] )


    # def make_pwc_matrix(self):
    #     """Matrix of pwc: C^O^N --> C^1"""
    #     A = np.zeros([int(self.dim_C1), int(self.dim_C0N)])

    #     for row in range(int(self.dim_C1)):
    #         edge = self.edges[row]
    #         i = utils.different_index(edge)

    #         minus_column = self.payoff_basis.index( (i, edge[0]) )
    #         plus_column = self.payoff_basis.index( (i, edge[1]) )
    #         A[row][minus_column] = -1
    #         A[row][plus_column] = +1

    #     return np.asmatrix(A)

    # def make_boundary_1_matrix(self):
    #     """Matrix of partial_1: C_1 --> C_0"""

    #     # Start with transpose
    #     A = np.zeros([int(self.dim_C1), int(self.dim_C0)])

    #     for row in range(int(self.dim_C1)):
    #         basis_edge = self.edges[row]

    #         minus_node, plus_node = basis_edge
    #         minus_column = self.nodes.index( minus_node )
    #         plus_column = self.nodes.index( plus_node )
    #         A[row][minus_column] = -1
    #         A[row][plus_column] = +1

    #     return A.transpose()

    # find cliques in graph
    # NB this does not work well, as it sometimes included duplicates that need be removed manually
    # better study and use nx.enumerate_all_cliques(G)
    # https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.clique.enumerate_all_cliques.html
    def k_cliques(self, graph):
        # 2-cliques
        cliques = [{i, j} for i, j in graph.edges() if i != j]
        k = 2

        while cliques:
            # result
            yield k, cliques

            # merge k-cliques into (k+1)-cliques
            cliques_1 = set()
            for u, v in combinations(cliques, 2):
                w = u ^ v
                if len(w) == 2 and graph.has_edge(*w):
                    cliques_1.add(tuple(u | w))

            # remove duplicates
            cliques = list(map(set, cliques_1))
            k += 1


    def print_cliques(self, graph, size_k):
        for k, cliques in self.k_cliques(graph):
            if k == size_k:
                print('%d-cliques = %d, %s.' % (k, len(cliques), cliques))

    def make_cliques(self, graph, size_k):
        for k, cliques in self.k_cliques(graph):
            if k == size_k:

                cliques = [list(clique) for clique in cliques]

                # edit
                # new_cliques = []
                # for elem in cliques:
                #     if elem not in new_cliques:
                #         new_cliques.append(elem)
                # cliques = new_cliques
                # end edit

                return cliques
               

    # def sort_oriented_simplices(self,list_of_simplices):
    #     for simplex in list_of_simplices:
    #         simplex.sort()

    def make_coboundary_1_matrix(self):
        """Matrix of d_1: C^1 --> C^2"""
        A = np.zeros([int(self.num_three_cliques), int(self.num_edges)])
        for row in range(int(self.num_three_cliques)):
            clique = self.three_cliques[row]
            i = self.edges.index([ clique[1], clique[2] ])
            j = self.edges.index([ clique[0], clique[2] ])
            h = self.edges.index([ clique[0], clique[1] ])
            A[row][i] = +1
            A[row][j] = -1
            A[row][h] = +1
        return np.asmatrix(A)

    def kill_duplicates(self, data):
        new_data = []
        for c in data:
            if c not in new_data:
                new_data.append(c)
        return new_data

    def check_duplicates(self, data):
        for i in range(len(data)):
            c = data[i]
            n = data.count(c)
            if n == 1:
                print(i, c, n)
            else:
                print(i, c, n, '<---')

    def draw_undirected_response_graph(self):

        G = self.networkx_graph

        draw_options = {

        'with_labels' : True,
        'font_weight' : 'bold',
        'arrows' : False,
        # 'node_color': 'blue',
        # 'node_size': 100,
        # 'width': 3,
        }

        nx.draw(G, **draw_options)
        plt.show()

    def find_maximal_cliques(self):
        return  nx.find_cliques(self.networkx_graph)

    def make_basis_harmonic_games(self, variables):
        """
        Returns
        - S = parametric form of kernel of harmonic matrix
        - B = basis of kernel of harmonic matrix
        """
        return sl.find_kernel(self.harmonic_matrix, variables)

    def make_random_harmonic_payoff_float(self):

        weights = np.random.uniform(0, 1, int(self.dim_H + self.dim_N))
        vectors = [ self.basis_harmonic_games[i] * weights[i] for i in range(int(self.dim_H + self.dim_N)) ]

        S = sum( vectors, sp.zeros(self.num_payoffs, 1))
        return(list(S))

    def make_random_harmonic_payoff_int(self):
        weights = np.random.randint(-3, 3+1, int(len(self.basis_harmonic_games)))
        vectors = [ self.basis_harmonic_games[i] * weights[i] for i in range(int(len(self.basis_harmonic_games))) ]

        # when summing, specify in second argument type of summand
        S = sum( vectors, sp.zeros(self.num_payoffs, 1))
        return(list(S))


    # def make_basis_strategic_harmonic_games(self):
    #     basis_strategic_harmonic_games = [e for e in self.basis_harmonic_games if e not in self.basis_nonstrategic_games]

    #     for e in basis_strategic_harmonic_games:
    #         print(f'e: {e}')
    #         print(f'pwc on e: {Matrix(self.pwc_matrix) * e}')
    #         print(f'harmonic on e: {Matrix(self.harmonic_matrix) * e}')
    #         print()
    #     return basis_strategic_harmonic_games
    ##########

    def pprint_parametric(self, variables, solution):
        """solution = solution of linear system in parametric form by Sympy, that is a FiniteSet object"""

        solution = list(solution)[0]
        assert(len(variables) == len(solution))

        for i in range(len(variables)):
            print(variables[i], '=', solution[i])
        print(utils.orange('------------'))


    def verbose_game_full(self):
        print(utils.red('\n-------------------- GAME FULL VERBOSE  -----------------------'))
        print('\n-------------------- GAME  -----------------------')
        print(f'N = {self.num_players}')
        print(f'NA = dim(C^0N) = {self.num_payoffs}')
        print(f'A = dim(C^0) = number of nodes = {self.num_strategy_profiles}')
        print(f'Basis of (C^0)^N, as many as NA = {self.payoff_basis}')
        print(f'Basis of C^0, as many as A = {self.nodes}')
        print(f'Basis of (C^1) = {self.edges}')
        print(f'Number of edges = dim(C^1) = {self.num_edges}')
        print(f'Number of 3-cliques = dim(C^2) = {self.num_three_cliques}')

        print('\n-------------------- CLUSTERS  -----------------------')
        for i in range(len(self.num_strategies_for_player)):
            Ai = self.num_strategies_for_player[i]
            num_i_clusters = int(self.num_nodes / Ai)
            print(f'For i = {i+1} there are {num_i_clusters} i-clusters, each with {Ai} nodes.' )


        print('\n-------------------- MAPS  -----------------------')
        print(f'pwc: {self.dim_C0N} --> {int(self.dim_C1)}')
        print(f'PWC Surjective = {self.pwc_is_surjective_numerical}')
        print(f'dim Im pwc = !dim closed! = {self.dim_im_pwc_theory}')
        print(f'dim Ker pwc = {self.dim_ker_pwc_theory}')
        print(f'd_0: {self.dim_C0} --> {int(self.dim_C1)}')
        print(f'd_1: {int(self.dim_C1)} --> {int(self.dim_C2)}')

        print('\n-------------------- C0N  -----------------------')
        print(f'dim(C^0N) = {self.num_payoffs}')
        print(self.payoff_basis)
        print('--------------------')
        print(f'dim N = dim Ker pwc = {self.dim_N}')
        print(f'dim P = dim Im d_0 = {self.dim_P}')
        print(f'dim H = !dim harmonic! {self.dim_H}')
        
        print('\n-------------------- C1  -----------------------')
        print(f'dim(C^1) = {self.dim_C1}')
        print(f'dim ker d_1 = dim closed = {self.dim_closed}')
        print('--------------------')
        print(f'codim closed = {self.dim_C1 - self.dim_closed}')
        print(f'dim exact = {self.dim_exact}')
        print(f'dim harmonic = {self.dim_harmonic}')

        print('\n-------------------- HARMONIC  -----------------------')
        print(f'harmonic (N+H) matrix: {int(self.dim_C0N)} --> {int(self.dim_C0)}')
        print(f'dim ker harmonic matrix = dim harmonic games (H + N) = {self.dim_ker_harmonic_matrix}\n')
        print(self.harmonic_matrix)
        print(f'dim im harmonic matrix = {self.dim_im_harmonic_matrix}')
        print(utils.orange('''\nCAREFUL! The parametric solution given by sympy returns a finiteset object, and the order of the solutions seems to be messed up sometimes.
        Do not rely on parametric solution, use basis solutions'''))
        print(utils.orange(f'Parametric form of harmonic games space = {self.harmonic_games_parametric}\n'))
        self.pprint_parametric(self.payoff_variables, (self.harmonic_games_parametric))
        print(f'\nBasis of harmonic games (H + N) = basis of kernel of harmonic matrix')
        [print(list(e.T)) for e in self.basis_harmonic_games]
        print('\nRandom harmonic payoff')
        if not self.harmonic_system_solved:
            print(utils.red('Unable to solve fully harmonic system'))
        print(self.random_harmonic_payoff)
        # print(f'\nBasis of nonstrategic games = {self.basis_nonstrategic_games}')
        print('\n-------------------- INNER PRODUCT  -----------------------')
        print('C0')
        print(self.metric_0.matrix)
        print('C1')
        print(self.metric_1.matrix)
        print()
        print(sp.latex(sp.Matrix(self.metric_0.matrix)))
        print(sp.latex(sp.Matrix(self.metric_1.matrix)))

