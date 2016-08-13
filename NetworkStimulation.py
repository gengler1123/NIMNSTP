from math import sqrt

class NetworkStimulation():
	"""Takes in a center point and generates a list of neuron indices within a certain radius of that point."""
	def __init__(self, Network, xCenter, yCenter, radius):
		self.Network = Network
		self.xCenter = xCenter
		self.yCenter = yCenter
		self.radius = radius

		self.nodes=[]

		for i in range(self.Network.number_of_nodes()):
			distance = sqrt((self.Network.node[i]["x"]-xCenter)**2 + \
				(self.Network.node[i]["y"]-yCenter)**2)
			if distance <= radius:
				self.nodes += [i]

	def getNodes(self):
		return self.nodes

