import numpy as np
import numpy.linalg as la
import scipy

########################################################
# Global methods
########################################################

def sqrt(pos_def_matrix):
	return scipy.linalg.sqrtm(pos_def_matrix)


def has_full_column_rank(matrix):
	"""matrix: a --> b"""
	b, a = np.shape(matrix)
	r = la.matrix_rank(matrix)
	return r == a

########################################################
# Metric generators
########################################################

class RandomMetric():
	"""input dimension, output metric"""
	def __init__(self, dim):
		self.matrix = self.make_random_metric(dim)

	def make_random_metric(self,n):

		# Generate random nxn matrix with entries uniformly distributed in (0,1)
		A = np.random.rand(n, n)

		# Get its transpose
		At = A.transpose()

		# For any invertible matrix A, the matrix A * At is symmetric and positive definite
		return np.matmul(A,At)

class EuclideanMetric():
	"""input dimension, output metric"""
	def __init__(self, dim):
		self.matrix = np.eye(dim)

class DiagonalMetric(object):
	"""input dimension, output metric"""
	def __init__(self, dim, N):

		self.matrix = np.eye(dim)

		for i in range(dim):
			self.matrix[i][i] = i + N
		
class ManualMetric():
	"""input manual matrix generator of metric, output metric"""
	def __init__(self, A):
		A = np.array(A)
		self.matrix = self.make_metric(A)

	def make_metric(self, A):
		At = A.transpose()
		return np.matmul(A,At)

########################################################
# Metric class
########################################################

class Metric():

	def __init__(self, matrix):
		"""Input metric as numpy matrix"""

		self.check_metric(matrix)

		self.dim = len(matrix)

		self.matrix = matrix
		self.inverse_matrix = self.inverse(self.matrix)

		self.flat_matrix = self.matrix
		self.sharp_matrix = self.inverse_matrix

	def check_metric(self, g):

		"""
		If a matrix A is symmetric (in general Hermitian) and positive definite then it admits a unique Cholesky decomposition,
		that is a decomposition A = L * Lt where L is a lower triangular matrix with real and positive diagonal entries, and Lt its conjugate transpose.

		If A is not Hermitian positive definite, the la.cholesky() method raises an error, so it can be used as a test for the input to be a metric.
		"""
		la.cholesky(g)


	def inverse(self, A):
		return la.inv(A)

	def sharp(self, covector):
		"""Returns vector given metric and covector via musical isomorphism
		These are equivalent:
		- sharp(u)
		- np.matmul(sharp_matrix, u)
		"""
		return np.einsum('ij,j', self.inverse_matrix, covector)

	def flat(self, vector):
		"""Returns covector given metric and vector via musical isomorphism
		These are equivalent:
		- flat(u)
		- np.matmul(flat_matrix, u)
		"""
		return np.einsum('ij,j', self.matrix, vector)

	def test(self):
		u = np.random.rand(self.dim)

		print(u)
		print(self.sharp(self.flat(u)))

		w = np.matmul(self.flat_matrix, u)
		z = np.matmul(self.sharp_matrix, w)
		print(z)

########################################################
# Homomorphism class
########################################################

class Homomorphism():
	"""Homomorphism between inner product spaces A: V --> W
	The two metrics are Metric instances"""
	def __init__(self, matrix, metric_domain, metric_codomain):

		self.matrix = matrix
		self.metric_domain = metric_domain
		self.metric_codomain = metric_codomain
		
		self.adjoint = self.make_adjoint()
		self.pinv = self.make_pinv()

	def make_adjoint(self):
		return self.metric_domain.sharp_matrix @ self.dual(self.matrix) @ self.metric_codomain.flat_matrix

	def dual(self, A):
		return np.transpose(A)

	def make_pinv(self):
		"""Choose method"""

		if has_full_column_rank(self.matrix):
			return self.make_pinv_full_col_rank_method()

		return self.make_pinv_square_root_method()

	def make_pinv_square_root_method(self):
		"""Remark 6 Kamaraj Sivakumur 2005. Always works."""
		N = self.metric_domain.matrix 
		M = self.metric_codomain.matrix
		A = self.matrix

		return la.inv(sqrt(N)) @ la.pinv( sqrt(M) @ A @ la.inv(sqrt(N))  ) @ sqrt(M)

	def make_pinv_full_col_rank_method(self):
		"""Works only if has_full_column_rank(self.matrix) == True
		and in this case equivalent to make_pinv
		but has nicer floating representation more often"""
		try:
			return la.inv((self.adjoint @ self.matrix)) @ self.adjoint
		except:
			raise Exception('Trying to feed matrix with linearly dependent columns to Homomorphism.make_pinv_full_col_rank_method method')


# A = np.array([[1, 1], [1, 0]])
# B = ManualMetric(A).matrix

# G = Metric(B)

# print(G.matrix)