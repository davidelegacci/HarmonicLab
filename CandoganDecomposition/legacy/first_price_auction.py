import numpy as np 


class SinglePriceAuction():
	"""Makes payoff of 2 players FPSB auction given n_discr for bids space and values for the two players"""
	def __init__(self, n_discr, values):

		'''
		i player index, ranges in self.players
		a bid index, ranges from 1 to effective_num_bids included
		'''

		self.n_discr = n_discr
		self.values = values
		self.num_players = len(values)
		self.players = range(self.num_players)

		self.step = 1 / (n_discr - 1)

		# Effective number of bids, determining game size
		self.skeleton = [ int(self.effective_num_bids(v)) for v in values ]

		self.bids = [[self.bid(a) for a in range(1, self.skeleton[i] + 1)] for i in self.players]

		# self.utility is the utility vector to feed to Candogan's code
		self.num_utilities = self.num_players * np.prod(self.skeleton)
		self.utility = [self.make_utility( i, [bid_0, bid_1]) for i in self.players for bid_0 in self.bids[0] for bid_1 in self.bids[1]]
		assert len(self.utility) == self.num_utilities

	def bid(self, a):
		'''Discretize bids space into n bids: [0, step, 2*step, ... (n_discr - 1) * step, n_discr * step = 1]
		a ranges from 1 to effective_num_bids included'''
		return (a - 1) * self.step

	def effective_num_bids(self, value):
		'''Effective number of bids to avoid negative utility'''
		return 1 + value * (self.n_discr - 1)

	def print_game_size(self):
		print(f'Single Price Auction: {self.skeleton} ')
		[print(f'Bids player {i}: {self.bids[i]}') for i in self.players]

	def make_utility(self, player, bids):
		'''
		bids = [bid_0, bid_1]
		bid_0 in self.bids_0
		bid_1 in self.bids_1
		output: 
		'''
		bid_player = bids[player]
		bid_opponent = bids[not(player)]

		if bid_player > bid_opponent:
			return self.values[player] - bid_player

		elif bid_player < bid_opponent:
			return 0

		else: return (self.values[player] - bid_player)/2

	def print_utility(self):
		print(f'Utility: {self.utility}')

# print('Single price auction')
# A = SinglePriceAuction(5,[1,0.8])
# A.print_game_size()
# A.print_utility()



