def generateSons(s):
	listOfSons = [] # armazena todos os filhos do estado s
	#listOfSons = list()

	# encher o pote 1
	state = [7, s[1]]
	listOfSons.append(state)

	# encher o pote 2
	state = [s[0], 5]
	listOfSons.append(state)

	# esvaziar o pote 1
	state = [0, s[1]]
	listOfSons.append(state)

	# esvaziar o pote 2
	state = [s[0], 0]
	listOfSons.append(state)

	# verter o pote 1 no pote 2
	if s[0] >= 5-s[1]:
		state = [s[0] - (5-s[1]), 5]
	else:
		state = [0, s[1] + s[0]]
	listOfSons.append(state)

	# verter o pote 2 no pote 1
	amountPossible = 7 - s[0]
	if amountPossible >= s[1]:
		state = [s[0] + s[1], 0]
	else:
		state = [7, s[1] - amountPossible]
	listOfSons.append(state)

	return listOfSons

def isGoal(s):
	if s[0] == 4 or s[1] == 4:
		return True
	else:
		return False


def son2str(s):
	return ''.join([str (v) for v in s])

def bfs(start):
	candidates = [start] # Fronteira de exploração
	fathers = dict()     # Armazena os pais dos nós no processo de busca
	visited = [start]    # Armazena a lista de nós visitados

	


























