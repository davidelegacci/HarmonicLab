'''
Symbolical computations for normal form game
- payoff and payoff field
- exact condition on reduced payoff field to verify iff potential
- coexact condition on reduced payoff field to verify iff harmonic

Contains also code to plot RD in 2x2 game, but from symbolical to numpy, not efficient;
for RD makes more sense to re-implement from skratch; done in RD_22.py
'''

import sympy as sp
import math as mt
import itertools
from pprint import pprint
import numpy as np
import utils
import time
from scipy.integrate import odeint
import matplotlib.pyplot as plt


##############################################################################################################################
##############################################################################################################################
############################################# SET GAME SKELETON ############################################# 

SKELETON = [2,2, 2, 3]

##############################################################################################################################
##############################################################################################################################

# Obsidian
open_tex = '$$'
close_tex = '$$'

# TexMaker
# open_tex = '\\['
# close_tex = '\\]'

'''
Gives symbolic expression of payoff and primitive 1-form of mixed extenson of normal form game
'''

##############################################################################################################################
##############################################################################################################################

class Player():
    """
    Player class. Each player has
    - name i
    - number of strategies Ai
    - strategies [1, 2, ..., Ai]
    """
    def __init__(self, num_strategies, player_name):
        self.num_strategies = num_strategies
        self.strategies = [ai for ai in range(0, num_strategies)]
        self.reduced_strategies = self.strategies[1:]
        self.player_name = player_name

##############################################################################################################################
##############################################################################################################################

class Payoff():
    """game is Game instance"""
    def __init__(self, game, payoff_vector = 0):
        """Either pass payoff vector as list, or generate random
        Integer payoff True by default; set false payoff can be float"""

        self.game = game

        if payoff_vector == 0:
            # generates sequential [0, 1, 2, ...]
            # payoff_vector = [i for i in range(self.game.NA)]
            payoff_vector = np.random.randint(-5, 5+1, int(self.game.NA))


        self.payoff_vector = payoff_vector
        self.payoff_vector_sympy = sp.Matrix(self.payoff_vector)

        self.payoff_dict = self.make_payoff_dict()

        self.u_payoff = [ ui.subs(self.payoff_dict) for ui in self.game.u_payoff ]
        self.u_payoff_pullback = [ ui.subs(self.payoff_dict) for ui in self.game.u_payoff_pullback ]

        self.V_field = [ Vi.subs(self.payoff_dict) for Vi in self.game.V_field ]
        self.V_components = [ Vi_ai.subs(self.payoff_dict) for Vi in self.game.V_components for Vi_ai in Vi  ]

        self.V_field_latex = self.make_V_field_latex()

        self.v_field = [ vi.subs(self.payoff_dict) for vi in self.game.v_field ]
        self.v_components = [ vi_ai.subs(self.payoff_dict) for vi in self.game.v_field_components for vi_ai in vi  ]
        self.v_field_latex = self.make_v_field_latex()

        self.V_exact_system = [expr.subs(self.payoff_dict) for expr in self.game.V_exact_system]
        self.v_exact_system = [expr.subs(self.payoff_dict) for expr in self.game.v_exact_system]

        self.delta_v = self.game.delta_v.subs(self.payoff_dict).simplify()
        self.two_delta_v_latex = utils.format_u(sp.latex((2 * self.delta_v).simplify()), self.game.players_names)

        self.payoff_verbose()


    def make_payoff_dict(self):
        """
        The output is a dictionary assigning the payoff to each player and strategy profile, taking as input elements of game.payoff_basis 
        So u[e] = u_i(a) for e = (i,a) in self.payoff_basis
        It is true that self.payoff_dict[e] == self.payoff_function[a][i] for e = (i,a)
        """
        # return dict(zip(self.game.payoff_basis, self.payoff_vector))
        try:
            assert self.game.NA == len(self.payoff_vector)
        except:
            print('Game skeleton and number of dofs provided do not match in size')
            print('''
                quick list for size of payoff_vector
                2x2:   AN = 8
                3x3:   AN = 18
                2x2x2: AN = 24
                4x4:   AN = 32
                ''')
            raise(Exception)

        return dict(zip(self.game.payoff_variables, self.payoff_vector))

    def mixed_strategy_dict(self, x):
        list_of_dicts =  [ dict(zip(self.game.mixed_strategy_profile[i], x[i])) for i in range(len(x)) ]
        unique_dict = {k: v for d in list_of_dicts for k, v in d.items()}
        return unique_dict


    def numeric_payoff(self, expr):
        '''Replaces utility symbol u_i(a) with number in any expression'''
        return expr.subs(self.payoff_dict)

    def numeric_mixed_strategy(self, expr, x):
        '''Replaces mixed strategy symbol x_i,a with number in any expression'''
        return expr.subs(self.mixed_strategy_dict(x))

    def u_bar(self, i, x):
        # returns number
        payoff_replaced = self.numeric_payoff(self.game.u_bar(i, self.game.mixed_strategy_profile))
        strategy_replaced = self.numeric_mixed_strategy(payoff_replaced, x)
        return strategy_replaced

    def vi(self, i, x):
        # takes as input player i = 1, 2, ... and mixed strategy, and returns list
        payoff_replaced = [self.numeric_payoff(el) for el in self.game.vi(i, self.game.mixed_strategy_profile)]
        strategy_replaced = [self.numeric_mixed_strategy(el, x) for el in payoff_replaced]
        return strategy_replaced

    def u_bar_check(self, i, x):
        '''i is player name, ranging 1, 2, ..., so need to slice x as x[i-1]'''
        return np.array( self.vi(i, x) ) @ np.array(x[i-1])

    def replicator(self, x, i, a):
        '''
        i player name, running 1, 2, ...
        a = pure strategy of player i, running 1, 2, ...
        i = Player.player_name
        a in Player.strategies
        '''
        
        player_index = i-1 # rename player to get index right
        action_index = a-1 # rename to get indexing right

        # e.g. 2x3 game
        # x = [ [0.2, 0.8], [0.5, 0.1, 0.4] ]
        # player 2, third strategy
        # 0.4 = x[2-1][3-1]
       
        xi_ai = x[player_index][action_index]
        vi = self.vi(i, x)
        vi_ai = vi[action_index]

        return xi_ai * (vi_ai - self.u_bar(i,x))

    def Replicator(self,x):
        update = []
        for i in range(1, len(x)+1):
            update_player = []
            player_index = i - 1
            for a in range(1, len( x[player_index] ) + 1 ):
                action_index = a-1
                update_player.append(self.replicator(x, i, a))
            update.append(update_player)
        return update

    def REPLICATOR(self, flat_x, t):
        x = utils.factor(flat_x, self.game.num_strategies_for_player)
        update = self.Replicator(x)
        return utils.flatten(update)

    def ode(self, x0, t_start, t_stop, N):
        print(f'Initial condition: {x0}')
        x0 = utils.flatten(x0)
        t = np.linspace(t_start, t_stop, N)
        data = odeint(self.REPLICATOR, x0, t)
        return data

    def extract_ode_data(self, x0, t_start, t_stop, N):
        data = self.ode(x0, t_start, t_stop, N)
        factored_data = [utils.factor(point, self.game.num_strategies_for_player) for point in data]
        first_strategy_for_each_player = [ [strategy[0] for strategy in timestep  ] for timestep in factored_data  ]
        return first_strategy_for_each_player

    def plot_replicator(self, X0, t_start, t_stop, N):
        start = time.time()
        if self.game.num_strategies_for_player == [2,2]:
            for x0 in X0:
                data = self.extract_ode_data(x0, t_start, t_stop, N)
                plt.plot(*zip(*data), 'k')
                x0_plot = data[0]
                plt.plot(*x0_plot, 'ro')
            ax = plt.gca()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            ax = plt.gca()
            ax.set_aspect('equal')#, adjustable='box')
            plt.show()
        elif self.game.num_strategies_for_player == [2,2,2]:
            data = self.extract_ode_data(x0, t_start, t_stop, N)
            x, y, z = utils.coords_points(data)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.grid()
            ax.plot3D(x, y, z)
            end = time.time()
            print("Execution time:",(end-start) * 10**3, "ms")
            plt.show()

    def make_V_field_latex(self):

        V = []

        for i in self.game.players_names:
            Vi_form = self.V_field[i-1]
            Vi_form_latex = utils.format_V( sp.latex(Vi_form), i )
            V.append(Vi_form_latex)
        return V

    def make_v_field_latex(self):
        v = []

        for i in self.game.players_names:
            vi_form = self.v_field[i-1]
            vi_form_latex = utils.format_v( sp.latex(vi_form), i )
            v.append(vi_form_latex)
        return v


    def make_test_mixed_strategy(self):
        return [ utils.make_mixed_strategy(p.num_strategies) for p in self.game.players ]



    def payoff_verbose(self):
        # print(f'\nNUMERIC PAYOFF: {self.payoff_vector}')
        # print(f'Payoff dict: {self.payoff_dict}')
        print('\n -------------------------------------------------------------------------------------- ')
        print('\n -------------------------------------------------------------------------------------- ')
        print('\n ----------- Begin numeric: specify payoff')
        print('\nPayoff dict:')
        print(open_tex)
        print('\\begin{split}')
        for key in self.payoff_dict:
            print_key = sp.latex(key)
            print_key = utils.format_u(print_key, self.game.players_names)
            print(print_key, '& =', self.payoff_dict[key], '\\\\')
        print('\\end{split}')
        print(close_tex)

        # Start numeric, works, comment off
        # test_mixed_strategy_x = self.make_test_mixed_strategy()
        # print(f'\nNUMERIC MIXED STRATEGY: {test_mixed_strategy_x}')
        # print(f'Mixed strategy dict: {self.mixed_strategy_dict(test_mixed_strategy_x)}\n')
        # for p in self.game.players:
        #     i = p.player_name
        #     print(f'\nPlayer {i}')
        #     print(f'u player {i} at mixed strategy profile {test_mixed_strategy_x} = {self.u_bar(i, test_mixed_strategy_x)}')
        #     print(f'u player {i} at mixed strategy profile {test_mixed_strategy_x} check = {self.u_bar_check (i, test_mixed_strategy_x)}')
        #     print(f'v player {i} at mixed strategy profile {test_mixed_strategy_x} = {self.vi(i, test_mixed_strategy_x)}')
        #     print()
        #     for a in p.strategies:
        #         print(f'replicator update for player {i} at mixed strategy profile {test_mixed_strategy_x} for pure strategy {a} = {self.replicator(test_mixed_strategy_x, i, a)}')
        # End numeric, works, comment off
        
        # print('\n-- REPLICATOR TEST UPDATE')
        # print(self.Replicator(test_mixed_strategy_x))
        # print(self.REPLICATOR(utils.flatten(test_mixed_strategy_x), t = 0))
        # print('\nODE test')
        # data = self.extract_ode_data( self.make_test_mixed_strategy(), t_start = 0, t_stop = 1, N = 3 )
        # print('--')
        # [print(o) for o in data]


        # Start symbolic, already included in Game class, works, comment off
        # for p in self.game.players:
        #     i = p.player_name
        #     print(f'\nbegin player {i}\n')
        #     print(f'u player {i} at mixed strategy profile {self.game.mixed_strategy_profile} = {self.game.u_bar(i, self.game.mixed_strategy_profile) }')
        #     print(f'v player {i} at mixed strategy profile {self.game.mixed_strategy_profile} = {self.game.vi(i, self.game.mixed_strategy_profile) }')
        # End symbolic, already included in Game class, works, comment off


        # for p in self.game.players:

        #     i = p.player_name

        #     print(f'{open_tex}  u^{{{i}}}(x) = {sp.latex(self.u_payoff[i-1])} {close_tex}\n')
        #     print(f'{open_tex} \\begin{{split}} V^{{{i}}}(x) & = {self.V_field_latex[i-1]} \\end{{split}}{close_tex}\n')

        print('\nPAYOFF AND PAYOFF FIELD numerical')
        for p in self.game.players:

            i = p.player_name
            print(f'Player {i}')

            # print(f'u^{i}(x) = {self.u_payoff[i-1]}\n')
            # print(f'V^{i}(x) = {self.V_field[i-1]}\n')
            # print(f'v^{i}(y) = {self.v_field[i-1]}\n')


            print(f'{open_tex}  u^{{{i}}}(x)  = {utils.format_u(sp.latex(self.u_payoff[i-1]), self.game.players_names)} {close_tex}\n')
            print(f'{open_tex}  u^{{{i}}}(y)  = {utils.format_u(sp.latex(self.u_payoff_pullback[i-1]), self.game.players_names)} {close_tex}\n')
            print(f'{open_tex} \\begin{{split}} V^{{{i}}}(x) & = {self.V_field_latex[i-1]} \\end{{split}}{close_tex}\n')
            print(f'{open_tex}  v^{{{i}}}(y)  = {self.v_field_latex[i-1]} {close_tex}\n')
        
        print('\n--- EXACT CONDITIONS numerical, each must be zero ----')
        print('\nV exact iff all these are zero')
        print(self.V_exact_system)
        print('\nv exact iff all these are zero')
        print(self.v_exact_system)

        print('\nCODIFFERENTIAL numerical')
        print(f'{open_tex}\\delta(v) = {self.two_delta_v_latex}{close_tex}')  

##############################################################################################################################


# If FORMAT == 'x' the variables that appear in the parametric solution of the harmonic system are x1, x2, ..., xAN
# Else, the variables are strings like u(i, (a)), explicitely showing player and strategy profile
FORMAT = 'u'
class Game():
    """
    Game class
    """
    def __init__(self, num_strategies_for_player):

        """num_strategies_for_player is list, e.g. [2,2] for 2 players 2 strategies each, etc.
        """

        self.num_strategies_for_player = num_strategies_for_player
        
        # Number of players
        self.N = len(num_strategies_for_player)

        # List of Player instances
        self.players = []
        for i in range(len(num_strategies_for_player)):
            self.players.append(Player(num_strategies_for_player[i], i+1))
            
        # Curly N, that is list of players names over which i runs
        self.players_names = [p.player_name for p in self.players] 

        # A, that is number of strategis profiles, that is number of nodes of the response graph
        self.A = int(mt.prod(num_strategies_for_player))

        # AN, that is dimension of payoff space
        self.NA = self.A * self.N

        # Curly_A set, that is set of strategies profiles, of cardinality A over which a runs
        # e.g. for 2x2 game it looks like [(1,1), (1,2), (2,1), (2,2)]
        self.pure_strategy_profiles = list(itertools.product(*[p.strategies for p in self.players]))

        # Basis of (C^0)^N of cardinality AN, i.e. basis of vector space of payoffs
        # Its elements are e = (i,a) for i in N for a in A
        # e.g. for 2x2 it looks like [[0, (1, 1)], [0, (1, 2)], [0, (2, 1)], [0, (2, 2)], [1, (1, 1)], [1, (1, 2)], [1, (2, 1)], [1, (2, 2)]]
        # self.payoff_basis = [ [i.player_name-1, a] for i in self.players for a in self.pure_strategy_profiles ]

        self.payoff_basis = [ (i, a) for i in self.players_names for a in self.pure_strategy_profiles ]
        self.payoff_strings = [ 'u'+str(e) for e in self.payoff_basis ]

        # for s in self.payoff_strings:
        #     assert(type(s) == str)

        self.payoff_variables = [sp.Symbol(s) for s in self.payoff_strings]

        # self.test_variables = []
        # for i in range(1, len(self.payoff_basis)+1):
        #     tmp_st = f'x{i}'
        #     globals()[tmp_st] = sp.Symbol(tmp_st)
        #     self.test_variables.append(globals()[tmp_st])

        # if FORMAT ==  'x':
        #     self.payoff_variables = self.test_variables

        self.mixed_strategy_profile = self.make_symbolic_mixed_strategy_profile()
        self.full_coordinates = [ xi_ai for xi in self.mixed_strategy_profile for xi_ai in xi ]

        self.reduced_mixed_strategy_profile = self.make_reduced_symbolic_mixed_strategy_profile()

        self.reduced_coordinates = [ yi_ai for yi in self.reduced_mixed_strategy_profile for yi_ai in yi ]

        # dict
        self.on_simplex = self.make_reduction_dict()


        self.basis_one_forms = self.make_basis_one_forms()
        self.reduced_basis_one_forms = self.make_reduced_basis_one_forms()

        self.u_payoff, self.u_payoff_latex = self.make_expected_payoff()

        self.u_payoff_pullback = [expr.subs(self.on_simplex).factor(self.reduced_coordinates) for expr in self.u_payoff]
        self.u_payoff_pullback_latex = [ utils.format_u( sp.latex( ui ), self.players_names ) for ui in self.u_payoff_pullback]

        self.V_field, self.V_field_latex, self.V_components = self.make_full_payoff_field()

        # List of 1-forms and list of corresponding components
        self.v_field, self.v_field_components = self.make_reduced_payoff_field()
        self.v_field_latex = self.make_v_field_latex()

        # Little hack to print field in single player case
        self.break_verbose()

        self.random_payoff_dict = { u:np.random.randint(-10, 10) for u in self.payoff_variables }


        # EXACT
        self.V_exact_system = self.make_V_exact_conditions()
        self.v_exact_system = self.make_v_exact_conditions()

        self.sol_dict_exact = utils.solve_system(self.v_exact_system, self.payoff_variables)


        self.random_exact_game = self.generate_random_exact_game()


        # CO-EXACT
        self.delta_v, self.two_delta_v_latex, self.delta_v_poly = self.compute_delta_v()

        '''
        self.delta_v_poly is the polynomial delta(v)
        with variables in reduced coordinates y
        and coefficients in payoff variables u.
        First extract coefficients, then solve linear system
        to generate a game fulfilling delta(v) = 0, i.e. to generate a co-exact game.
        '''
        self.sol_dict_coexact = utils.solve_poly_system(self.delta_v_poly, self.payoff_variables)

        self.random_coexact_game = self.generate_random_coexact_game()

        self.verbose_game()
        self.print_random_games()

        # SHA METRIC ## NOT GOOD, TO DO; CF sha_metric_on_simplex.py
        # self.g_sha = self.make_g_sha()

    #################################################################################################
    # BEGIN METHODS
    #################################################################################################

    def is_pure_strategy_profile(self,a):
        try:
            assert a in self.pure_strategy_profiles
        except:
            raise Exception(f'Invalid pure strategy profile {a}')


    def is_numeric_mixed_strategy_profile(self,x):
        '''x is list of lists: x = [x_1, ..., x_N], length = num players
        Each xi is probabiolity distribution, list xi = [xi_1, ..., xi_Ai], length = num pure strategies of player i'''

        # Length of mixed strategy profile = number of players, i.e. there is one mixed strategy xi per player
        try:
            assert len(x) == self.N
        except:
            raise Exception(f'{len(x)} mixed strategies provided but {self.N} needed')

        # Each mixed strategy xi is indeed probability distribution of the right size
        i = 1
        for xi in x:
            player = self.players[i-1]

            # Adds up to 1
            try:
                assert sum(xi)==1
            except:
                raise Exception(f'Invalid mixed strategy profile {xi}, does not add up to 1')

            # Length of mixed strategy = number of pure strategies of player i
            try:
                assert len(xi) == player.num_strategies
            except:
                raise Exception(f'Invalid mixed strategy profile {xi}, must have {player.num_strategies} entries but has {len(xi)}')

            # No negative components
            for xi_ai in xi:
                try:
                    assert xi_ai >= 0
                except:
                    raise Exception(f'Invalid mixed strategy profile {xi}, some entry is negative')

            i +=1


    def is_symbolic_mixed_strategy_profile(self,x):
        '''x is list of lists: x = [x_1, ..., x_N], length = num players
        Each xi is probabiolity distribution, list xi = [xi_1, ..., xi_Ai], length = num pure strategies of player i'''

        # Length of mixed strategy profile = number of players, i.e. there is one mixed strategy xi per player
        try:
            assert len(x) == self.N
        except:
            raise Exception(f'{len(x)} mixed strategies provided but {self.N} needed')

        # Each mixed strategy xi is indeed probability distribution of the right size
        i = 1
        for xi in x:
            player = self.players[i-1]


            # Length of mixed strategy = number of pure strategies of player i
            try:
                assert len(xi) == player.num_strategies
            except:
                raise Exception(f'Invalid mixed strategy profile {xi}, must have {player.num_strategies} entries but has {len(xi)}')

            i +=1

    def is_mixed_strategy_profile(self, x):

        symbolic = False
        for xi in x:
            for xi_ai in xi:
                if isinstance(xi_ai, sp.Expr):
                    symbolic = True
                    break

        if symbolic:
            self.is_symbolic_mixed_strategy_profile(x)
        else:
            self.is_numeric_mixed_strategy_profile(x)

    def make_mixed_strategy(self, i):
        '''Returns symbolic mixed strategy'''

        try:
            assert i in self.players_names
        except:
            raise Exception(f'Invalid player {i}')

        xi = []
        player = self.players[i-1]
        for ai in player.strategies:
            #tmp_st = f'x{i}_{ai}'
            tmp_st = f'\\x{{{i}}}{{{ai}}}'
            globals()[tmp_st] = sp.Symbol(tmp_st)
            xi.append(globals()[tmp_st])

        return xi

    def make_symbolic_mixed_strategy_profile(self):
        '''Returns symbolic mixed strategy profile'''
        x = [ self.make_mixed_strategy(i) for i in self.players_names ]
        self.is_symbolic_mixed_strategy_profile(x)
        return x

    def make_reduced_mixed_strategy(self, i):
        '''Returns symbolic mixed strategy'''

        try:
            assert i in self.players_names
        except:
            raise Exception(f'Invalid player {i}')

        yi = []
        player = self.players[i-1]
        Ai = player.num_strategies
        for ai in player.reduced_strategies:
            #tmp_st = f'x{i}_{ai}'
            tmp_st = f'\\y{{{i}}}{{{ai}}}'
            globals()[tmp_st] = sp.Symbol(tmp_st)
            yi.append(globals()[tmp_st])

        return yi

    def make_reduced_symbolic_mixed_strategy_profile(self):
        '''Returns symbolic mixed strategy profile'''
        y = [ self.make_reduced_mixed_strategy(i) for i in self.players_names ]
        return y

    def make_reduction_dict(self):
        on_simplex = {  }
        for p in self.players:
            i = p.player_name
            xi = self.mixed_strategy_profile[i-1]
            yi = self.reduced_mixed_strategy_profile[i-1]
            on_simplex[ xi[0] ] = 1 - sum(yi)
            for ri in p.reduced_strategies:
                on_simplex[ xi[ri] ] = yi[ri-1]
        return on_simplex
            

    def make_basis_one_forms_i(self, i):

        try:
            assert i in self.players_names
        except:
            raise Exception(f'Invalid player {i}')

        dxi = []
        player = self.players[i-1]
        Ai = player.num_strategies
        for ai in player.strategies:
            # tmp_st = f'dx_{i}^{ai}'
            tmp_st = f'\\dxx{{{i}}}{{{ai}}}'
            globals()[tmp_st] = sp.Symbol(tmp_st)
            dxi.append(globals()[tmp_st])

        return dxi

    def make_basis_one_forms(self):
        dx = [ self.make_basis_one_forms_i(i) for i in self.players_names ]
        return dx

    def make_reduced_basis_one_forms_i(self, i):

        try:
            assert i in self.players_names
        except:
            raise Exception(f'Invalid player {i}')

        dyi = []
        player = self.players[i-1]
        Ai = player.num_strategies
        for ai in player.reduced_strategies:
            # tmp_st = f'dx_{i}^{ai}'
            tmp_st = f'\\dyy{{{i}}}{{{ai}}}'
            globals()[tmp_st] = sp.Symbol(tmp_st)
            dyi.append(globals()[tmp_st])

        return dyi

    def make_reduced_basis_one_forms(self):
        dy = [ self.make_reduced_basis_one_forms_i(i) for i in self.players_names ]
        return dy


    def u(self,i,a):
        '''Returns sympy symbol for payoff of player i at pure strategy profile a'''
        try:
            assert i in self.players_names
        except:
            raise Exception(f'Invalid player {i}')
        try:
            assert a in self.pure_strategy_profiles
        except:
            raise Exception(f'Invalid strategy {a}')
        s = 'u'+str( (i, a) )
        return sp.Symbol(s)

    def p(self,x,a):
        '''Returns probability that pure strategy profile a occurs given mixed strategy profile x
        x is tuple of probability distributions [xi] 
        x can be both symbolic or numeric mixed strategy profile'''
        self.is_pure_strategy_profile(a)
        self.is_mixed_strategy_profile(x)



        L = [  x[i-1][ a[i-1]  ] for i in self.players_names  ]

        return sp.prod(L)

    def u_bar(self, i, x):
        L = [self.u(i,a) * self.p(x,a) for a in self.pure_strategy_profiles]
        return sum(L)

    def vi_ai(self, i, ai, x):

        # Prevent over writing symbolic mixed strategy profile
        x = x.copy()

        try:
            assert i in self.players_names
        except:
            raise Exception(f'{i} is not a valid player')
        self.is_mixed_strategy_profile(x)

        player = self.players[i-1]
        try:
            assert(ai in player.strategies)
        except:
            raise Exception(f'{ai} is not a valid pure strategy for player {i}')

        # Replace xi with certain ai [0, .., 0, 1, 0, .., 0]
        xi_new = [0 for _ in range(player.num_strategies) ]
        xi_new[ai ] = 1
        x[i-1] = xi_new
        # Now x contains [0, .., 0, 1, 0, .., 0] in the right position so v can be evaluated with u_bar
        return self.u_bar(i, x)

    def vi(self, i, x):
        player = self.players[i-1]
        return [self.vi_ai(i, ai, x) for ai in player.strategies]

    def v(self,x):
       
        return [self.vi(i, x) for i in self.players_names ]


    def show_probabilities(self, x):
        '''Given mixed strategy profile x (numeric or symbolic) shows probability for each pure strategy profile a'''
        print(f'Mixed strategy profile: {x}')
        print('Pure profile : probability')
        for a in self.pure_strategy_profiles:
            print( a,':', self.p( x, a ) )


    def make_expected_payoff(self):
    
        u = [ self.u_bar(p.player_name, self.mixed_strategy_profile) for p in self.players ]

        u_latex = [ utils.format_u( sp.latex( ui ), self.players_names ) for ui in u]

        return (u, u_latex)
        

    def make_full_payoff_field(self):

        V = [  ]
        V_components = [  ]
        V_latex = [  ]

        for p in self.players:

            i = p.player_name

            Vi_components = self.vi(i, self.mixed_strategy_profile)
            Vi_form = list(sp.Matrix(Vi_components).T * sp.Matrix(self.basis_one_forms[i-1]))[0] # Sympy Vi form
            
            Vi_form_latex = utils.format_V( sp.latex(Vi_form), i )

            V.append(Vi_form)
            V_latex.append(Vi_form_latex)
            V_components.append(Vi_components)

        return V, V_latex, V_components

    def make_reduced_payoff_field(self):
        v_components = []
        v = [ ]
        for p in self.players:
            i = p.player_name
            Vi_components = self.V_components[i-1]
            vi_components = [Vi_components[r] - Vi_components[0] for r in p.reduced_strategies]
            vi_components = [expr.subs(self.on_simplex).factor(self.reduced_coordinates).simplify() for expr in vi_components]
            v_components.append(vi_components)
            vi_form = list(sp.Matrix(vi_components).T * sp.Matrix(self.reduced_basis_one_forms[i-1]))[0] # Sympy Vi form
            v.append(vi_form)
        return v, v_components

    def make_V_field_latex(self):

        V = []

        for i in self.players_names:
            Vi_form = self.V_field[i-1]
            Vi_form_latex = utils.format_V( sp.latex(Vi_form), i )
            V.append(Vi_form_latex)
        return V

    def make_v_field_latex(self):
        v = []

        for i in self.players_names:
            vi_form = self.v_field[i-1]
            vi_form_latex = utils.format_v( sp.latex(vi_form), i )
            v.append(vi_form_latex)
        return v

    def diff_full_payoff_field(self, i, ai, j, bj):
        '''
        i, j = players
        ai, bj = strategy indices = 0, 2, ...
        returns d( v_{i, ai} ) / d( y_{j, bj} ) 
        '''
        V = self.V_components
        # Player starts at 1 so index must be i-1
        Vi_ai = V[ i-1 ][ ai  ]

        x = self.mixed_strategy_profile
        xj_bj = x[ j-1 ][ bj ]
        return sp.diff(Vi_ai, xj_bj)

    def diff_reduced_payoff_field(self, i, ai, j, bj):
        '''
        i, j = players
        ai, bj = REDUCED strategy indices = 1, 2, ...
        returns d( v_{i, ai} ) / d( y_{j, bj} ) 
        '''
        v = self.v_field_components
        # Player starts at 1 so index must be i-1
        # Reduced strategy starts at 1 so index must be ai - 1
        vi_ai = v[ i-1 ][ ai - 1 ]

        y = self.reduced_mixed_strategy_profile
        yj_bj = y[ j-1 ][ bj - 1 ]
        return sp.diff(vi_ai, yj_bj)

    def make_V_exact_conditions(self):
        '''V closed iff each is zero'''
        conditions = []
        for p in self.players:
            for q in self.players:
                for ai in p.strategies:
                    for bj in q.strategies:
                        i = p.player_name
                        j = q.player_name
                        C = self.diff_full_payoff_field(i, ai, j, bj) - self.diff_full_payoff_field(j, bj, i, ai)
                        if C != 0 and -C not in conditions:
                            conditions.append(C)
        return conditions

    # def make_v_exact_conditions_two_players(self):
    #     '''v closed iff each is zero'''
    #     conditions = []
    #     for p in self.players:
    #         for q in self.players:
    #             for ai in p.reduced_strategies:
    #                 for bj in q.reduced_strategies:
    #                     i = p.player_name
    #                     j = q.player_name
    #                     C = self.diff_reduced_payoff_field(i, ai, j, bj) - self.diff_reduced_payoff_field(j, bj, i, ai)
    #                     if C != 0 and -C not in conditions:
    #                         conditions.append(C)
    #     return conditions

    def make_v_exact_conditions(self):
        '''v closed iff each is zero'''
        conditions = []
        for p in self.players:
            for q in self.players:
                for ai in p.reduced_strategies:
                    for bj in q.reduced_strategies:
                        i = p.player_name
                        j = q.player_name
                        exact_poly_coefficients = sp.poly(self.diff_reduced_payoff_field(i, ai, j, bj) - self.diff_reduced_payoff_field(j, bj, i, ai), self.reduced_coordinates).coeffs()
                        for coeff in exact_poly_coefficients:
                            if coeff != 0 and -coeff not in conditions:
                                conditions.append(coeff)

        return conditions

    def compute_delta_v(self):

        delta_v_tuple = []

        V = self.V_components

        for p in self.players:

            i = p.player_name
            Ai = p.num_strategies

            ui = self.u_payoff[ i-1 ] # number
            Vi = V[ i-1 ] # list of components

            delta_v_tuple.append(Ai * ui - sum(Vi))

        # Key: delta_v is a polynomial
        # with reduced coordinates y as variables
        # and payoffs u as coefficients
        delta_v = sum(delta_v_tuple).subs(self.on_simplex).factor(self.reduced_coordinates).simplify() / 2

        # Write as poly to extract coefficients
        delta_v_poly = sp.poly(delta_v, self.reduced_coordinates)

        # Latex formatting
        two_delta_v_latex = utils.format_u(sp.latex((2 * delta_v).simplify()), self.players_names)
        return delta_v, two_delta_v_latex, delta_v_poly

    def generate_random_coexact_game(self):
        random_coexact_payoff = { u : self.sol_dict_coexact[u].subs(self.random_payoff_dict) for u in self.sol_dict_coexact }
        return random_coexact_payoff

    def generate_random_exact_game(self):
        random_exact_payoff = { u : self.sol_dict_exact[u].subs(self.random_payoff_dict) for u in self.sol_dict_exact }
        return random_exact_payoff


    ### NOT GOOD ####
    # def make_g_sha(self):
    #     print(self.reduced_mixed_strategy_profile)
    #     g = [ ]
    #     for p in self.players:
    #         i = p.player_name
    #         yi = self.reduced_mixed_strategy_profile[i-1]
    #         # dummy_1, dummy_2 = sp.symbols('dummy_1, dummy_2')
    #         gi = sp.FunctionMatrix( p.num_strategies-1, p.num_strategies-1, 'lambda (dummy_1,dummy_2), sp.KroneckerDelta(dummy_1,dummy_2)  ')
    #         print(gi.as_explicit())
    #         # for ai in p.reduced_strategies:
    #         #     for bi in p.reduced_strategies:
    #         #         first_index = ai - 1
    #         #         second_index = bi - 1
    #         #         if first_index == second_index:
    #         #             gi.row(first_index).col(second_index) = yi[ai] - yi[ai]**2
    #         #         else:
    #         #             gi.row(first_index).col(second_index) = - yi[ai] * yi[bi]




    
    def verbose_game(self):
        print(f'Game = {self.num_strategies_for_player}')
        print(f'N = {self.N}')
        print(f'Players = {self.players_names}')
        print(f'Number of pure strategy profiles = {self.A}')
        print(f'Pure strategy profiles = {self.pure_strategy_profiles}')
        print(f'DOFS to specify payoff = {self.NA}')
        # print(f'Payoff basis = {self.payoff_basis}')
        print(f'\nPayoff variables = {self.payoff_variables}')
        
        print(f'Full coordinates = {self.full_coordinates}')
        
        print(f'Reduced coordinates = {self.reduced_coordinates}')
        print(f'\nBasis 1-forms = {self.basis_one_forms}')
        print(f'Reduced basis 1-forms = {self.reduced_basis_one_forms}')


        print(f'{open_tex}x = {self.mixed_strategy_profile}{close_tex}')
        print(f'{open_tex}y = {self.reduced_mixed_strategy_profile}{close_tex}')

        print('\nPAYOFF AND PAYOFF FIELD')
        for p in self.players:

            i = p.player_name
            print(f'Player {i}')

            # print(f'u^{i}(x) = {self.u_payoff[i-1]}\n')
            # print(f'V^{i}(x) = {self.V_field[i-1]}\n')
            # print(f'v^{i}(y) = {self.v_field[i-1]}\n')

            print(f'{open_tex}  u^{{{i}}}(x)  = {self.u_payoff_latex[i-1] } {close_tex}\n')
            print(f'{open_tex}  u^{{{i}}}(y)  = {self.u_payoff_pullback_latex[i-1] } {close_tex}\n')
            # print(self.u_payoff_pullback)
            print(f'{open_tex} \\begin{{split}} V^{{{i}}}(x) & = {self.V_field_latex[i-1]} \\end{{split}}{close_tex}\n')
            print(f'{open_tex}  v^{{{i}}}(y)  = {self.v_field_latex[i-1]} {close_tex}\n')

            # print('\nSwitch following off if slow')
            # print(f'Check: must be zero so that u^i(x) = V^i(x) cdot x^i: { (list(sp.Matrix(self.V_components[i-1]).T @ sp.Matrix(self.mixed_strategy_profile[i-1]))[0] - self.u_payoff[i-1]).simplify()}\n' )

        print('\nEXACT SYSTEM, each must be zero')
        print('\nV exact iff all these are zero')
        # print(self.V_exact_system)
        for eq in self.V_exact_system:
            print (f'{open_tex} {utils.format_u(sp.latex(eq), self.players_names)} = 0 {close_tex}')

        print('\nv exact iff all these are zero')
        # print(self.v_exact_system) 
        for eq in self.v_exact_system:
            print (f'{open_tex} {utils.format_u(sp.latex(eq), self.players_names)} = 0 {close_tex}')

        print('\nSolution of exact system for reduced payoff')
        utils.latex_print_dict(self.sol_dict_exact, self.players_names, open_tex, close_tex)

        print('\nRandom exact game')
        utils.latex_print_dict(self.random_exact_game, self.players_names, open_tex, close_tex)
        print(list(self.random_exact_game.values()))

        print('\n---------------------')

        print('\nCODIFFERENTIAL polynomial')
        print(f'{open_tex}2 \\delta(v) = {self.two_delta_v_latex}{close_tex}')

        print('\nCollect coefficients of delta(v), set each to zero, and solve linear system to generate co-exact game:')
        utils.latex_print_dict(self.sol_dict_coexact, self.players_names, open_tex, close_tex)

        print('\nRandom co-exact game')
        utils.latex_print_dict(self.random_coexact_game, self.players_names, open_tex, close_tex)
        print(list(self.random_coexact_game.values()))

    def print_random_games(self):
        
        print(f'\nRandom exact {self.num_strategies_for_player} game')
        utils.latex_print_dict(self.random_exact_game, self.players_names, open_tex, close_tex)
        print(list(self.random_exact_game.values()))


        print(f'\nRandom co-exact {self.num_strategies_for_player} game')
        utils.latex_print_dict(self.random_coexact_game, self.players_names, open_tex, close_tex)
        print(list(self.random_coexact_game.values()))


    def break_verbose(self):
        pass
        # self.verbose_game()

##############################################################################################################################
##############################################################################################################################

print('begin')
G = Game(SKELETON)


# quick list for size of payoff_vector
# 2x2:   AN = 8
# 3x3:   AN = 18
# 2x2x2: AN = 24
# 4x4:   AN = 32

# # Matching pennies
# u = Payoff(game = G, payoff_vector = [1, 0, 0, 1, 0, 1, 1, 0] )

# u = Payoff(game = G, payoff_vector = [-1/3, -6.0, 1.0, 2.0, -1/3, 1.0, -6.0, 2.0] )
# u = Payoff(game = G, payoff_vector = [-2. , -2. , -2. , -2. ,  2. ,  2. ,  2. ,  2. , -1. , -1. ,
#           1. ,  1. , -1. , -1. ,  1. ,  1. , -0.5,  0.5, -0.5,  0.5,
#          -0.5,  0.5, -0.5,  0.5] )
# u = Payoff(game = G, payoff_vector = [-7.00000000000000, 4.00000000000000, 5.00000000000000, -5.00000000000000, -3, 1, 2, -3, 8.00000000000000, -4.00000000000000, 1, 2, 0, -2, 2, -3, -6.00000000000000, -3, 1, -3, -2, 0, 3, 2] )

# pv = [-2.6875, -0.5625, -1.1875, 4.4375, 2.6875, 0.5625, 1.1875, -4.4375, 1.6875, 1.5625, -1.6875, -1.5625, 0.6875, -3.9375, -0.6875, 3.9375, 1.0, -1.0, 2.875, -2.875, -3.375, 3.375, -0.5, 0.5]

# pv = list(np.random.randint(-10, 10, 18))

# pv = [1, 1, 0, 0, 0, 1, 0, 0]

# PD = [2, 0, 3, 1, 2, 3, 0, 1] # Prisoner's dilemma
# MP = [3, 0, 0, 3, 0, 3, 3, 0] # Matching pennies

# u = Payoff(game = G, payoff_vector = MP)


print('\nend')

# Not working
# u.quiver_replicator(quiver_density = 10)
# end not working


# REPLICATOR OK
# print('start ode')
# u.plot_replicator(X0 = [u.make_test_mixed_strategy() for _ in range(1)], t_start = 0, t_stop = 5, N = 50)
# END REPLICATOR






