import numpy as np 


class SinglePriceAuction():
	"""2 players"""
	def __init__(self, max_num_bids, values):

		'''
		i player index, ranges in self.players
		a bid index, ranges from 1 to effective_num_bids included
		'''

		self.max_num_bids = max_num_bids
		self.values = values
		self.num_players = len(values)
		self.players = range(self.num_players)

		self.step = 1 / (max_num_bids - 1)

		# Effective number of bids, determining game size
		self.effective_num_bids = [ int(self.effective_num_bids(v)) for v in values ]

		self.bids = [[self.bid(a) for a in range(1, self.effective_num_bids[i] + 1)] for i in self.players]

		# self.utility is the utility vector to feed to Candogan's code
		self.num_utilities = self.num_players * np.prod(self.effective_num_bids)
		self.utility = [self.make_utility( i, [bid_0, bid_1]) for i in self.players for bid_0 in self.bids[0] for bid_1 in self.bids[1]]
		assert len(self.utility) == self.num_utilities

	def bid(self, a):
		'''Discretize bids space into n bids: [0, step, 2*step, ... (max_num_bids - 1) * step, max_num_bids * step = 1]
		a ranges from 1 to effective_num_bids included'''
		return (a - 1) * self.step

	def effective_num_bids(self, value):
		'''Effective number of bids to avoid negative utility'''
		return 1 + value * (self.max_num_bids - 1)

	def print_game_size(self):
		print(f'Single Price Auction: {self.effective_num_bids} ')
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



