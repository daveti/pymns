# pyMNS.py
# Python Markov Network Solver
# pyMNS reads in an uai file with description of a Markov Network
# and outputs the partition function for this MN.
# Ref: http://www.cs.huji.ac.il/project/UAI10/fileFormat.php
# Version 0.4
# 1. Enlarged system recursion limit for t4.uai
# June 5, 2013
# Version 0.3
# 1. Added loopy Belief Propagation
# May 30, 2013
# Version 0.2
# 1. Added Variable Elimination algorithm
# 2. Added min-neighbors heuristic
# 3. Added support for evidence file
# 4. Added support for MAP inference
# 5. Added other ordering heuristics
# May 12, 2013
# Version 0.1
# Input: *.uai
# Output: partition function
# May 1, 2013
# daveti@cs.uoregon.edu
# http://daveti.blog.com

import os
import sys
import math
import copy
from counter import Counter

# Debug flag
debugFlag = False

# Sys recursion limit setting
# NOTE: default value is 1000
# for t4.uai, we need to enlarge this value
sys.setrecursionlimit(20000)

# Define the factor data structure
class factor:
	def __init__(self):
		self.numOfVar = 0
		self.numOfVal = 0
		self.indexOfVar = ()
		self.cardinalityOfVar = ()
		self.strideOfVar = ()
		self.factTable = []

	def computeNumOfValue(self):
		'''
		Compute the numOfVal based on cardinalityOfVar
		'''
		sigma = 1
		for c in self.cardinalityOfVar:
			sigma *= c
		self.numOfVal = sigma

	def computeCardinality(self, card):
		'''
		Compute the cardinality based on indexOfVar and card
		'''
                cardinality = []
                for v in self.indexOfVar:
                        cardinality.append(card[v])
                self.cardinalityOfVar = tuple(cardinality)

	def computeStride(self):
		'''
		Compute the stride based on numOfVar and cardinalityOfVar
		'''
                if self.numOfVar == 1:
                        self.strideOfVar = tuple([1]) 
                else:
                        stride = []
                        for i in range(self.numOfVar-1):
                                sigma = 1 
                                for j in range(i+1, self.numOfVar):
                                        sigma *= self.cardinalityOfVar[j]
                                # Save this card
                                stride.append(sigma)
                        # Fix the last stride
                        stride.append(1)
                        self.strideOfVar = tuple(stride)

	def postProcess(self, card):
		'''
		Post process the factor to generate the cardinality and stride
		'''
		# Get the cardinality
		self.computeCardinality(card)

		# Compute the stride
		self.computeStride()

	def getStride(self, varIndex):
		'''
		Get the stride for certain variable within this factor
		NOTE: will return 0 if this variable is not contained
		'''
		for i in range(len(self.indexOfVar)):
			if self.indexOfVar[i] == varIndex:
				return(self.strideOfVar[i])
		return(0)
	
	def normalize(self):
		'''
		Normalize the factor to make it a probability distribution
		'''
		sum = 0.0
		for v in self.factTable:
			sum += v
		for i in range(self.numOfVal):
			self.factTable[i] /= sum

	def verify(self):
		'''
		Verify the internal consistency of this factor
		'''
		isConsistent = True
		isCardinality = False
		if self.numOfVar != len(self.indexOfVar):
			print('Error: fc_numOfVar, fc_indexOfVar')
			isConsistent = False
		if self.numOfVal != len(self.factTable):
			print('Error: fc_numOfVal, fc_factTable')
			isConsistent = False
		if self.numOfVar != len(self.cardinalityOfVar):
			print('Error: fc_numOfVar, fc_cardinalityOfVar')
			isConsistent = False
		for i in range(self.numOfVar):
			if self.cardinalityOfVar[i] != mn[mn_cardinality][self.indexOfVar[i]]:
				print('Error: fc_cardinality, mn_cardinality: ', i)
				isConsistent = False
				isCardinality = True
		if isCardinality == True:
			print('indexOfVar: ', self.indexOfVar)
			print('cardinality: ', self.cardinalityOfVar)
		return(isConsistent)
			
	def printFC(self):
		'''
		Print the factor
		'''
		print('fc_numOfVar: ', self.numOfVar)
		print('fc_numOfVal: ', self.numOfVal)
		print('fc_indexOfVar: ', self.indexOfVar)
		print('fc_cardinalityOfVar: ', self.cardinalityOfVar)
		print('fc_strideOfVar: ', self.strideOfVar)
		print('fc_factTable: ', self.factTable)
		print('len(factTable): ', len(self.factTable))

class cliqueFactor:
	'''
	Clique-Factor class definition used to help VE
	NOTE: the index of cliques and factors should be the same!
	'''
	def __init__(self, cliques, factors):
		'''
		Init the cliqueFactor object with cliques and factors from mn
		'''
		self.bitMap = Counter()
		self.cliques = cliques[:]
		self.factors = factors[:]

	def getFactorsFromVar(self, var):
		'''
		Get the factors involving with this variable
		'''
		output = []
		for i in range(len(self.cliques)):
			if self.bitMap[i] == 1:
				# This is the clique used before
				continue
			else:
				# Bypass the num of var within the clique
				for j in range(1, len(self.cliques[i])):
					if self.cliques[i][j] == var:
						# Got the factor
						output.append(self.factors[i])
						break
		return(output)

	def updateCliqueFactor(self, fact, var):
		'''
		Update the clique-factor object with new factor and elimited variable
		'''
		# Mark the old cliques with the eliminated var to be used
		for i in range(len(self.cliques)):
			if self.bitMap[i] == 1:
				# Old clique marked before
				continue
			else:
				# Bypass the num of var within the clique
				for j in range(1, len(self.cliques[i])):
					if self.cliques[i][j] == var:
						# Mark this clique used
						self.bitMap[i] = 1
						break
	
		# Construct the new clique
		newCli = []
		newCli.append(fact.numOfVar)
		newCli += list(fact.indexOfVar)

		# Add the new clique and factor
		self.cliques.append(tuple(newCli))
		self.factors.append(fact)

	def printCF(self):
		'''
		Print the clique-factor object
		'''
		print('cf_bitmaps: ', self.bitMap)
		print('cf_cliques: ', self.cliques)
		print('cf_factors: ', self.factors)

class message:
	'''
	Class definition for message used by Belief Propagation
	'''
	def __init__(self, varIdx, factIdx, isVar2Fact=False):
		self.varIdx = varIdx
		self.factIdx = factIdx
		self.isVar2Fact = isVar2Fact
		self.oldFact = None
		self.fact = None
		self.initValue = 1
		self.msgTable = []
		self.sumTable = []

	def printID(self):
		'''
		Print the identification for this msg for debug
		'''
		print('msg_varIdx: ', self.varIdx)
		print('msg_factIdx: ', self.factIdx)
		print('msg_isVar2Fact: ', self.isVar2Fact)

	def printMsg(self):
		'''
		Print the msg
		'''
		print('msg_varIdx: ', self.varIdx)
		print('msg_factIdx: ', self.factIdx)
		print('msg_isVar2Fact: ', self.isVar2Fact)
		print('msg_oldFact: ', self.oldFact)
		if self.oldFact != None:
			print('oldFactBegin---------------')
			self.oldFact.printFC()
			print('oldFactEnd---------------')
		print('msg_fact: ', self.fact)
		if self.fact != None:
			print('factBegin------------------')
			self.fact.printFC()
			print('factEnd------------------')
		print('msg_initValue: ', self.initValue)
		print('msg_msgTable: ', self.msgTable)
		print('msgTableBegin------------------')
		for m in self.msgTable:
			m.printID()
		print('msgTableEnd--------------------')
		print('msg_sumTable: ', self.sumTable)


# Define the Markov Network data structure
# mn['name'] = 'MARKOV'
# mn['numOfVar'] = 3
# mn['cardinality'] = (2, 2, 3)
# mn['numOfCliq'] = 3
# mn['cliques'] = [(1, 0), (2, 0, 1), (2, 1, 2)]
# mn['factors'] = [fac0, fac1, fac2...]
# Evidences added....
# mn['numOfEvid'] = 1
# mn['numOfEvidVar'] = 2
# mn['evidences'] = [(1, 0), (2, 1)]
mn_name = 'name'
mn_numOfVar = 'numOfVar'
mn_cardinality = 'cardinality'
mn_numOfCliq = 'numOfCliq'
mn_cliques = 'cliques'
mn_factors = 'factors'
mn_numOfEvid = 'numOfEvid'
mn_numOfEvidVar = 'numOfEvidVar'
mn_evidences = 'evidences'
mn = {}
mn[mn_cliques] = []
mn[mn_factors] = []
mn[mn_evidences] = []

# Define the Markove Network Factor Graph
# mn_fg_vf[(0, 1)] = M_var0->factor1(var0)
# mn_fg_fv[(0, 1)] = M_factor0->var1(var1)
mn_fg_vf = {}
mn_fg_fv = {}

# Helper functions here
def logFun(x, base):
	'''
	Log function log(base)x
	base could be 2, 10 or math.e
	'''
	if x == 0:
		return(0)
	else:
		return(math.log(x, base))

def debugFun(var1, var2):
	'''
	Debug function
	'''
	if debugFlag == True:
		print('Debug: ', var1, var2)

def verifyMN():
	'''
	Verify the consistency of MN
	'''
	num = 0
	print('Debug-verifyMN----------------')
	for f in mn[mn_factors]:
		if f.verify() == False:
			print('Error: Inconsistency within MN')
			f.printFC()
			num += 1
	if num != 0:
		print('Error: total %d inconsistency' %(num))
	print('Debug-verifyMN----------------')
			
def printMN():
	'''
	Dump all the information within the mn data structure
	'''
	print('mn_name: ', mn[mn_name])
	print('mn_numOfVar: ', mn[mn_numOfVar])
	print('mn_cardinality: ', mn[mn_cardinality])
	print('mn_numOfCliq: ', mn[mn_numOfCliq])
	print('mn_cliques: ', mn[mn_cliques])
	print('len(mn_factors): ', len(mn[mn_factors]))
	for f in mn[mn_factors]:
		f.printFC()
	print('mn_numOfEvid: ', mn[mn_numOfEvid])
	print('mn_numOfEvidVar: ', mn[mn_numOfEvidVar])
	print('mn_evidences: ', mn[mn_evidences])

def printMNFG():
	'''
	Dump all the information for mn factor graph
	'''
	print('=================================V2F: ', len(mn_fg_vf))
	print('mn_fg_vf: ', mn_fg_vf)
	for i in mn_fg_vf:
		mn_fg_vf[i].printMsg()
		print('-----------------------------')
	print('=================================F2V: ', len(mn_fg_fv))
	print('mn_fg_fv: ', mn_fg_fv)
	for i in mn_fg_fv:
		mn_fg_fv[i].printMsg()
		print('-----------------------------')
		
# Define globle parsing variables helping parseing
parseLineForFactor = False
parseUntill= 0
parseFactorIndex = 0
def parseUAILine(line, index):
	'''
	Parse the UAI line into the mn data structure
	'''
	global parseLineForFactor
	global parseUntill
	global parseFactorIndex

	if index == 0:
		# Save the name
		mn[mn_name] = line
	elif index == 1:
		# Save the number of variables
		mn[mn_numOfVar] = int(line)
	elif index == 2:
		# Save the cardinalities
		card = line.split()
		card2 = []
		for c in card:
			card2.append(int(c))
		mn[mn_cardinality] = tuple(card2)
	elif index == 3:
		# Save the number of cliques
		mn[mn_numOfCliq] = int(line)
		# Mark the parsing flag
		parseUntill = index + mn[mn_numOfCliq]
	else:
		if parseLineForFactor == False:
			# Parse cliques in progress
			cliques = line.split()
			cliques2 = []
			for c in cliques:
				cliques2.append(int(c))
			mn[mn_cliques].append(tuple(cliques2))

			# Create a factor for this clique record
			fact = factor()
			fact.numOfVar = cliques2[0]
			fact.indexOfVar = tuple(cliques2[1:])
			mn[mn_factors].append(fact)

			if parseUntill == index:
				parseLineForFactor = True
				parseUntill = 0
		else:
			# Parse the factors in progress
			# NOTE: need to handle 2 cases here
			# (A)
			# 2
			# 0.99 0.01
			# (B)
			# 2 0.99 0.01
			if parseUntill == 0:
				# Try to find the format
				lineSp = line.split()
				if len(lineSp) == 1:
					# Format A
					mn[mn_factors][parseFactorIndex].numOfVal = int(line)
					parseUntill = int(line)
				else:
					# Format B
					mn[mn_factors][parseFactorIndex].numOfVal = int(lineSp[0])
					for i in range(1, len(lineSp)):
						mn[mn_factors][parseFactorIndex].factTable.append(float(lineSp[i]))
                                        # Finish processing this factor
                                        mn[mn_factors][parseFactorIndex].postProcess(mn[mn_cardinality])
                                        # Move to the next factor
					parseFactorIndex += 1

			else:
				val = line.split()
				for v in val:
					mn[mn_factors][parseFactorIndex].factTable.append(float(v))
			
				if len(mn[mn_factors][parseFactorIndex].factTable) == parseUntill:
					# Finish processing this factor
					mn[mn_factors][parseFactorIndex].postProcess(mn[mn_cardinality])
					# Move to the next factor
					parseUntill = 0
					parseFactorIndex += 1 


def parseUAIEvidLine(line, index):
        '''
        Parse the UAI Evidence line into the mn data structure
        '''
        if index == 0:
                # Save the numOfEvid
                mn[mn_numOfEvid] = int(line)
        elif index == 1:
                # Parse and save the evidence line
		evid = line.split()
		mn[mn_numOfEvidVar] = int(evid[0].strip())
		evidVarList = []
		evidVarVal = []
		for i in range(1, len(evid)):
			# Add this value into varVal pair
			evidVarVal.append(int(evid[i].strip()))
			if i % 2 == 0:
				# Add the last varVal pair
				evidVarList.append(tuple(evidVarVal))
				# Start of a new evid var
				evidVarVal = []
		# Add this evidences into the mn
		mn[mn_evidences] = evidVarList
	else:
		print('Error: more than 1 line of evidence')


def loadParseUAIFile(filePath, isEvidence=False):
	'''
	Load the UAI file and parse it into Markov Network data structure
	NOTE: this function is extended to handle the evidence file too.
	'''
	if not os.path.exists(filePath):
		print('Error: %s does not exists' % (filePath))
	else:
		try:
			fileObj = open(filePath, "r")
			ii = 0
			for line in fileObj:
				line = line.strip()
				# Bypass the empty lines
				if line == '':
					continue
				# Parse each valid line
				if isEvidence == False:
					parseUAILine(line, ii)
				else:
					parseUAIEvidLine(line, ii)
				ii += 1
		finally:
			fileObj.close()

# Define functions for constructing factor graph in MN
def initFactorGraph4MN():
	'''
	Init the factor graph (mn_fg_vf and mn_fg_fv) with keys
	'''
	for i in range(mn[mn_numOfCliq]):
	# i is the index of factors
		for j in range(1, len(mn[mn_cliques][i])):
			mn_fg_vf[(mn[mn_cliques][i][j], i)] = message(mn[mn_cliques][i][j], i, True)
			mn_fg_fv[(i, mn[mn_cliques][i][j])] = message(mn[mn_cliques][i][j], i)

def constructVar2FactorMsg():
	'''
	Construct the message from var to factor
	'''
	for k in mn_fg_vf:
		var = mn_fg_vf[k].varIdx
		fact = mn_fg_vf[k].factIdx
		for i in range(mn[mn_numOfCliq]):
		# i is the index of factors
			if i != fact:
			# Exclude the to-factor itself
				for j in range(1, len(mn[mn_cliques][i])):
				# j is the index of vars
					# Find the factors connected with this var except this factor
					if mn[mn_cliques][i][j] == var:
						# Add this factor2var message into the list
						mn_fg_vf[k].msgTable.append(mn_fg_fv[(i, var)])
						break

def constructFactor2VarMsg():
	'''
	Construct the message from factor to var
	'''
	for k in mn_fg_fv:
		var = mn_fg_fv[k].varIdx
		fact = mn_fg_fv[k].factIdx
		# Get the cligue from the factIdx
		for j in range(1, len(mn[mn_cliques][fact])):
			# Exclude the var itself
			if var != mn[mn_cliques][fact][j]:
				# Add this var2factor message into the list
				mn_fg_fv[k].msgTable.append(mn_fg_vf[(mn[mn_cliques][fact][j], fact)])
				# Add this var into summing out table
				mn_fg_fv[k].sumTable.append(mn[mn_cliques][fact][j])

def checkDiff(oldFact, newFact, variance):
	'''
	Check if the difference between oldFact and newFact within the variance
	Return: True/if it is; False otherwise
	'''
	# Defensive checking
	if oldFact.numOfVal != newFact.numOfVal:
		print('Error: checkDiff')
		return(False)

	output = True
	for i in range(oldFact.numOfVal):
		if abs(oldFact.factTable[i] - newFact.factTable[i])/oldFact.factTable[i] > variance:
			output = False
			break
	return(output)

def product4BP(msg1, msg2):
	'''
	Message product for Belief Propagation
	Input: message structure
	Output: factor structure
        '''
	output = None
	if msg1.fact != None and msg2.fact != None:
		output = product(msg1.fact, msg2.fact, mn[mn_cardinality])
	elif msg1.fact != None:
		output = copy.deepcopy(msg1.fact)
	elif msg2.fact != None:
		output = copy.deepcopy(msg2.fact)
	
	if output != None:
		output.normalize()
	return(output)
		
def sum4BP(msg):
	'''
	Sum out the var for this message
	Input: message structure
	Output: none
	'''
	if msg.fact != None:
		for v in msg.sumTable:
			msg.fact = marginalize(v, msg.fact)

def checkConverge4BP(msgVariance, maxLoop, currentLoop):
	'''
	Check if the BP is converged or not based on the msg variance and max loop number
	Output: True/False
	'''
	if currentLoop >= maxLoop:
		return(True)
	# Check the variance only after the 2nd loop
	# when we could have different and non-none oldFact and fact
	if currentLoop > 2:
		# Compute the variance of all msgs
		for vf in mn_fg_vf:
			if mn_fg_vf[vf].fact == None:
				# No update for this msg
				continue
			else:
				# Compare the oldFact and fact
				if checkDiff(mn_fg_vf[vf].oldFact, mn_fg_vf[vf].fact, msgVariance) == False:
					return(False)
		for fv in mn_fg_fv:
			if mn_fg_fv[fv].fact == None:
				# No update for this msg
				continue
			else:
				# Compare the oldFact and fact
				if checkDiff(mn_fg_fv[fv].oldFact, mn_fg_fv[fv].fact, msgVariance) == False:
					return(False)
		return(True)
	return(False)

def loopyBeliefPropagation():
	'''
	Loopy Belief Propagation
	'''
	# Init the factor graph and message passing
	initFactorGraph4MN()
	constructVar2FactorMsg()
	constructFactor2VarMsg()
	#printMNFG()

	loopNum = 0
	# Fix the condition of convergence here
	while checkConverge4BP(0.0001, 50, loopNum) == False:
		# Update the msg from var to factor
		if loopNum != 0:
			# Bypass the first update as init value will be used
			for vf in mn_fg_vf:
				# Debug:
				'''
				print('Debug: ', mn_fg_vf[vf])
				mn_fg_vf[vf].printMsg()
				'''
				if len(mn_fg_vf[vf].msgTable) == 0:
					# var-->factor
					# No update for this msg
					continue
				elif len(mn_fg_vf[vf].msgTable) == 1:
					# Save the old msg
					mn_fg_vf[vf].oldFact = copy.deepcopy(mn_fg_vf[vf].fact)
					# Update the msg without production
					mn_fg_vf[vf].fact = copy.deepcopy(mn_fg_vf[vf].msgTable[0].fact)
					if mn_fg_vf[vf].fact != None:
						mn_fg_vf[vf].fact.normalize()
				else:
					# Save the old msg - do not need deep copy here
					mn_fg_vf[vf].oldFact = mn_fg_vf[vf].fact
					# Compute the msg with production
					# Debug:
					'''
					print('msg1--------------')
					mn_fg_vf[vf].msgTable[0].printMsg()
					print('msg2--------------')
					mn_fg_vf[vf].msgTable[1].printMsg()
					print('daveti: ')
					daveti = product(mn_fg_vf[vf].msgTable[0].fact, mn_fg_vf[vf].msgTable[1].fact, mn[mn_cardinality])
					daveti.printFC()
					'''
					mn_fg_vf[vf].fact = product4BP(mn_fg_vf[vf].msgTable[0], mn_fg_vf[vf].msgTable[1])
					'''
					if mn_fg_vf[vf].fact == None:
						print('daveti: product4BP2 does not work!')
					'''
					for f in range(2, len(mn_fg_vf[vf].msgTable)):
						mn_fg_vf[vf].fact = product4BP(mn_fg_vf[vf].msgTable[f], mn_fg_vf[vf])

                # Debug
		'''
                print('loopNum: ', loopNum)
                if loopNum == 1:
			printMNFG()
                        break
		'''

		# Update the msg from factor to var
		for fv in mn_fg_fv:
			# Save the old msg
			mn_fg_fv[fv].oldFact = copy.deepcopy(mn_fg_fv[fv].fact)
			# Always product the general factor at first
			mn_fg_fv[fv].fact = copy.deepcopy(mn[mn_factors][mn_fg_fv[fv].factIdx])
			# Update the msg
			for f in range(len(mn_fg_fv[fv].msgTable)):
				mn_fg_fv[fv].fact = product4BP(mn_fg_fv[fv].msgTable[f], mn_fg_fv[fv])
			# Sum out the var
			sum4BP(mn_fg_fv[fv])
			# Normalize the factor
			mn_fg_fv[fv].fact.normalize()

		# Update the loop num
		loopNum += 1
		# Debug
		#print('loopNum: ', loopNum)

	# Debug
	#print('BP-loopNum: ', loopNum)

def computeMarginal4BP():
	'''
	Compute the marginal probability after Belief Propagation
	'''
	for i in range(mn[mn_numOfVar]):
		print(i)
		msg = None
		for fv in mn_fg_fv:
			if mn_fg_fv[fv].varIdx == i:
				if msg == None:
					msg = copy.deepcopy(mn_fg_fv[fv])
				else:
					msg.fact = product4BP(msg, mn_fg_fv[fv])
		print(msg.fact.factTable)
				

# Define functions for operation in MN
def getVarAssignmentFromFactorIndex(fact, index):
	'''
	Get the variable assignment based on the index within the factor
	Get the variable configuration for the entry within the factor
	output: variable assginment list
	Reference: K&F P359
	'''
	varAssList = []
	for i in range(fact.numOfVar):
		assgin = index / fact.strideOfVar[i] % fact.cardinalityOfVar[i]
		varAssList.append(assgin)
	return(varAssList)

def product(fact1, fact2, card):
	'''
	Multiply 2 factors and output a new factor
	Reference: Probabilistic Graphical Model - K&F P359
	'''
	# Defensive checking
	if fact1.verify() == False:
		print('Error: product')
		fact1.printFC()
		return(None)
	if fact2.verify() == False:
		print('Error: product')
		fact2.printFC()
		return(None)

	output = factor()
	# Process the output factor
	# Generate all other info for the factor based on
	# indexOfVar of the 2 factors and cardinality in MN
	X1 = set(fact1.indexOfVar)
	X2 = set(fact2.indexOfVar)
	X1UX2 = X1.union(X2)
	x1ux2List = []
	for x in X1UX2:
		x1ux2List.append(x)
	x1ux2List.sort()
	output.indexOfVar = tuple(x1ux2List)
	output.numOfVar = len(x1ux2List)
	output.postProcess(card)
	output.computeNumOfValue()
	
	# Efficient implementation of a factor product operation
	j = 0
	k = 0
	assignment = {}
	# Hack
	uaiOrderOfIndex = list(output.indexOfVar)
	uaiOrderOfIndex.reverse()

	for l in output.indexOfVar:
		assignment[l] = 0
	for i in range(output.numOfVal):
		output.factTable.append(fact1.factTable[j]*fact2.factTable[k])
		# NOTE: hack here
		# for l in output.indexOfVar:
		# Original algorithm assign the values starting from the first variable
		# However, UAI file format needs different value assignment starting
		# from the last variable. Above commented for loop would generate the
		# original (flat) layout suggested by the K&F; the for loop down below
		# will follow the UAI (flat) layout!
		for l in uaiOrderOfIndex:
			assignment[l] += 1
			if assignment[l] == card[l]:
				# We reach the last possible value of this var
				assignment[l] = 0
				j -= (card[l]-1)*fact1.getStride(l)
				k -= (card[l]-1)*fact2.getStride(l)
			else:
				j += fact1.getStride(l)
				k += fact2.getStride(l)
				break
	# Defensive checking again to make sure nothing wrong within this funciton
	if output.verify() == False:
		print('Error: product - cause inconsistency')

	return(output)

def marginalize(var, fact):
	'''
	Marginalize the factor for a var and return a new factor
	'''
	# Defensive checking
	if fact.verify() == False:
		print('Error: marginalize')
		fact.printFC()
		return(None)

	output = factor()
	# Defensive checking
	if var not in fact.indexOfVar:
		print('Error: no such a variable %d within the factor' %(var))
		return(None)

	# Debug
	'''
	print('var: ', var)
	fact.printFC()
	'''

	# Preconstruct the new output factor
	output.numOfVar = fact.numOfVar - 1
	index = fact.indexOfVar.index(var)
	#print('index: ', index)
	tempList = list(fact.indexOfVar)
	#tempList.remove(tempList[index])
	tempList.pop(index)
	#print('tempList: ', tempList)
	output.indexOfVar = tuple(tempList)
	tempList = list(fact.cardinalityOfVar)
	#tempList.remove(tempList[index])
	tempList.pop(index)
	#print('tempList: ', tempList)
	output.cardinalityOfVar = tuple(tempList)
	output.computeNumOfValue()
	output.computeStride()

	'''
	print('Precheck the construting...')
	output.verify()
	print('Precheck done...')
	'''

	# Debug
	if debugFlag == True:
		output.printFC()

	if fact.numOfVar == 1:
		# This is the final table
		sum = 0
		for v in fact.factTable:
			sum += v
		output.factTable.append(sum)
	else:
		# Construct the new factor table
		bitMap = Counter()
		pairOffset = fact.strideOfVar[index]
		loopNum = fact.cardinalityOfVar[index]
		for i in range(fact.numOfVal):
			sum = 0
			# Bypass the cross-out'd entry
			if bitMap[i] == 1:
				continue
			else:
				sum = fact.factTable[i]
				pairIdx = i
				for l in range(loopNum-1):
					# NOTE: i+pairOffset should be valid!
					pairIdx += pairOffset
					pairIdx %= fact.numOfVal
					# Sum up
					sum += fact.factTable[pairIdx]
					# Mark the bitmap for pair wise entry
					bitMap[pairIdx] = 1
				# Add this summing out
				output.factTable.append(sum)

	# Defensive checking & debuging
	if len(output.factTable) != output.numOfVal:
		print('Error: bad marginalization')

	# Defensive checking again to make sure nothing wrong within this funciton
	if output.verify() == False:
		print('Error: marginalize - cause inconsistency')
	return(output)

def getNeighborsOfVar(cliques, var):
	'''
	Get the neighbors of the var
	'''
	neighbors = set()
	for c in cliques:
		for i in range(1, len(c)):
			if c[i] == var:
				# All the vars within the cliques are neighbors except var
				for j in range(1, len(c)):
					neighbors.add(c[j])
				break
	# Remove the var itself
	neighbors.remove(var)
	# Debug
	#print('neighbors: ', var, neighbors)
	return(list(neighbors))

def getWeightOfVar(cliques, var):
	'''
	Get the weights of the var
	'''
	sigma = 1
	neighbors = getNeighborsOfVar(cliques, var)
	for v in neighbors:
		sigma *= mn[mn_cardinality][v]
	return(sigma)

def isTheEdgeContainedInGraph(cliques, edge):
	'''
	Check if this edge is contained in the graph
	'''
	for c in cliques:
		vertexL = False
		vertexR = False
		for i in range(1, len(c)):
			if c[i] == edge[0]:
				vertexL = True
			elif c[i] == edge[1]:
				vertexR = True
			if vertexL == True and vertexR == True:
				return(True)
	return(False)
			
def getEdgesNeededOfVar(cliques, var):
	'''
	Get the edges needed to eliminate this var
	'''
	edges = set()
	neighbors = getNeighborsOfVar(cliques, var)
	# Construct the edge between any 2 neighbors
	for v1 in neighbors:
		for v2 in neighbors:
			if v1 != v2:
				edges.add((v1, v2))
	# Eliminate all the existing edges within the graph
	edges = list(edges)
	for e in edges:
		if isTheEdgeContainedInGraph(cliques, e) == True:
			edges.remove(e)
	return(edges)

def getFillOfVar(cliques, var):
	'''
	Get the fill of the var
	'''
	return(len(getEdgesNeededOfVar(cliques, var)))

def getWeightedFillOfVar(cliques, var):
	'''
	Get the weighted fill of the var
	'''
	edges = getEdgesNeededOfVar(cliques, var)
	sum = 0
	for e in edges:
		sum += mn[mn_cardinality][e[0]]*mn[mn_cardinality][e[1]]
	return(sum)

def updateGraph(cliques, var):
	'''
	Update the graph to eliminate the var and add a new clique
	'''
	neighbors = set()
	for c in cliques:
		for i in range(1, len(c)):
			if c[i] == var:
				# Add all the neighbors into the set
				for j in range(1, len(c)):
					neighbors.add(c[j])
				cliques.remove(c)
				break
	# Remove the var itself
	neighbors.remove(var)
	# Add the new clique
	neighbors = list(neighbors)
	clique = [len(neighbors)]
	clique += neighbors
	cliques.append(tuple(clique))

def minNeighbors(cliques, varList):
	'''
	Heuristic function for min-neighbors
	input: the graph model with cliques
	output: the index of the variable
	'''
	minOfNei = len(getNeighborsOfVar(cliques, varList[0]))
	var = varList[0]
	for i in range(1, len(varList)):
		neighbors = getNeighborsOfVar(cliques, varList[i])
		lenOfNei = len(neighbors)
		if lenOfNei < minOfNei:
			minOfNei = lenOfNei
			var = varList[i]
	return(var)

def minWeight(cliques, varList):
	'''
	Heuristic function for min-weight
	'''
	minOfWei = getWeightOfVar(cliques, varList[0])
	var = varList[0]
	for i in range(1, len(varList)):
		wei = getWeightOfVar(cliques, varList[i])
		if wei < minOfWei:
			minOfWei = wei
			var = varList[i]
	return(var)

def minFill(cliques, varList):
	'''
	Heuristic function for min-fill
	'''
	minOfFill = getFillOfVar(cliques, varList[0])
	var = varList[0]
	for i in range(1, len(varList)):
		fill = getFillOfVar(cliques, varList[i])
		if fill < minOfFill:
			minOfFill = fill
			var = varList[i]
	return(var)

def weightedMinFill(cliques, varList):
	'''
	Heuristic function for weighted min fill
	'''
	minOfWeiFill = getWeightedFillOfVar(cliques, varList[0])
	var = varList[0]
	for i in range(1, len(varList)):
		weiFill = getWeightedFillOfVar(cliques, varList[i])
		if weiFill < minOfWeiFill:
			minOfWeiFill = weiFill
			var = varList[i]
	return(var)
			
# Global mapping for heuristics
mnHeu = {}
mnHeu[0] = 'min-neighbors'
mnHeu[1] = 'min-weight'
mnHeu[2] = 'min-fill'
mnHeu[3] = 'weighted-min-fill'

def greedyOrdering(heu):
	'''
	Greedy search for constructing an elimination ordering
	Reference: K&F P314
	input: heuristic number
	output: an ordered list
	'''
	order = []
	cliques = mn[mn_cliques][:]
	varLeft = range(mn[mn_numOfVar])
	var = -1
	for i in range(mn[mn_numOfVar]-1):
		if heu == 0:
			var = minNeighbors(cliques, varLeft)
		elif heu == 1:
			var = minWeight(cliques, varLeft)
		elif heu == 2:
			var = minFill(cliques, varLeft)
		elif heu == 3:
			var = weightedMinFill(cliques, varLeft)
		else:
			print('Error: unsupported heuristics %d' % (heu))

		# Debug
		#print('min-var: ', var, varLeft)
		# Add this var into the order
		order.append(var)
		# Remove this var from varLeft
		varLeft.remove(var)
		# Update the cliques
		# Introduce edges in graph between all neighbors of var
		updateGraph(cliques, var)
	
	# Add the final var left
	order.append(varLeft[0])
	return(order)
			
def computePR4MN():
	'''
	Compute the partition function for this mn
	'''
	# NOTE: no heuristics added right now...
	output = None
	numOfFactor = len(mn[mn_factors])
	card = mn[mn_cardinality]
	if numOfFactor == 1:
		# We have got only 1 factor
		output = mn[mn_factors][0]
	else:
		# We have got at least 2 factors
		output = product(mn[mn_factors][0], mn[mn_factors][1], card)
		if numOfFactor > 2:
			for i in range(2, numOfFactor):
				output = product(output, mn[mn_factors][i], card)
	# Debug
	if debugFlag == True:
		output.printFC()

	# Compute the PR
	sigma = 0
	for v in output.factTable:
		sigma += v
	return(sigma)

def computeVE4MN(var, factList, noMargin=False):
	'''
	Compute the VE ~ product all the factors and sum up the variable
	output: the new factor after ve
	'''
	output = None
	numOfFact = len(factList)
	card = mn[mn_cardinality]
	# Defensive checking
	if numOfFact == 0:
		print('Error: empty fact list')
	elif numOfFact == 1:
		if noMargin == False:
			# Just sum up this var
			output = marginalize(var, factList[0])
		else:
			output = factList[0]
	else:
		# At least 2 factors involveds
		output = product(factList[0], factList[1], card)
		if numOfFact > 2:
			for i in range(2, numOfFact):
				output = product(output, factList[i], card)

		if noMargin == False:
			output = marginalize(var, output)

	return(output)
	
def variableElimination(order=[], noMargin=False):
	'''
	Variable elimination algorithm framework
	input: (elimination) order (list)
	output: partition function if noMargin==False
		final joint table/factor if noMargin==True
	'''
	output = None
	# Construct the clique-factor management object
	cliFacMan = cliqueFactor(mn[mn_cliques], mn[mn_factors])
	if len(order) == 0:
		# Default ordering starting from 0
		order = range(mn[mn_numOfVar])

	# Go thru each var with VE
	# Debug
	# print('order-in-ve: ', order)
	for i in order:
		# Get the corresponding factors
		factList = cliFacMan.getFactorsFromVar(i)
		# Debug
		if debugFlag == True:
			for f in factList:
				f.printFC()
				print('------------------------------------')
		# Run VE
		output = computeVE4MN(i, factList, noMargin)
		# Update the management object
		cliFacMan.updateCliqueFactor(output, i)
		# Debug
		if debugFlag == True:
			print('new-factor:')
			output.printFC()
			print('==========================================')
			print('var: ', i)
			cliFacMan.printCF()

	# Debug
	if debugFlag == True:
		output.printFC()

	if noMargin == False:
		# Return the PR
		return(output.factTable[0])
	else:
		# Return this factor
		return(output)

def getTheSummingOutVars():
	'''
	Get the vars needs to be summing out
	output: var list
	'''
	varList = []
	# Construct the evidence var list
	evidVarList = []
	for e in mn[mn_evidences]:
		evidVarList.append(e[0])
	# Construt the summing out var list
	for v in range(mn[mn_numOfVar]):
		isEvidVar = False
		for ev in evidVarList:
			if ev == v:
				isEvidVar = True
				break
		if isEvidVar == False:
			# This is the var to be summing out
			varList.append(v)
	return(varList)

def computePROB4MNWithEvid(order=[]):
	'''
	Compute the probability of the evidences
	Based on VE
	'''
	# Defensive checking
	if len(mn[mn_evidences]) == 0:
		return(-1)

	output = None
	# Construct the clique-factor management object
	cliFacMan = cliqueFactor(mn[mn_cliques], mn[mn_factors])
	if len(order) == 0:
		# Default ordering starting from 0 and exlucde the evidence vars
		order = getTheSummingOutVars()

	# Go thru each var with VE
	# Debug
	if debugFlag == True:
		print('order-in-ve: ', order)
	leftFactors = mn[mn_factors][:]
	for i in order:
		# Get the corresponding factors
		factList = cliFacMan.getFactorsFromVar(i)
		# Debug
		if debugFlag == True:
			for f in factList:
				f.printFC()
				print('------------------------------------')
		# Run VE
		output = computeVE4MN(i, factList)
		# Update the management object
		cliFacMan.updateCliqueFactor(output, i)
		# Remove the used factors and add the new output factor
		for f in factList:
			leftFactors.remove(f)
		leftFactors.append(output)
		# Debug
		if debugFlag == True:
			print('new-factor:')
			output.printFC()
			print('==========================================')
			print('var: ', i)
			cliFacMan.printCF()

	# Get the joint table for left vars - the evidence vars
	# NOTE: we could probably think about the order here too...
	# Multiply all the left factors
        numOfFactor = len(leftFactors)
        card = mn[mn_cardinality]
        if numOfFactor > 1:
                # We have got at least 2 factors
                output = product(leftFactors[0], leftFactors[1], card)
                if numOfFactor > 2:
                        for i in range(2, numOfFactor):
                                output = product(output, leftFactors[i], card)

	# Debug
	if debugFlag == True:
		output.printFC()

	# Now we have got the final joint distribution table with evidence vars
	# Sum out all the entries within the factor table
	sum = 0
	for f in output.factTable:
		sum += f

	# Get the entry value for the evidence
	fIndex = 0
	for e in mn[mn_evidences]:
		# Get the index of this evidence var
		eIndex = output.indexOfVar.index(e[0])
		# Get the stride of this evidence var
		eStride = output.strideOfVar[eIndex]
		# Move the index based on the stride of the var
		fIndex += e[1] * eStride
	# Get the final value
	evid = output.factTable[fIndex]

	# Return the prob
	return(float(evid)/sum)

def computeMAP4MN(order=[]):
	'''
	Compute the MAP(MPE) inference for the non-evidence variable
	Based on VE
	output: the most probable configuration for the non-evidence vars
	'''
	jointTable = variableElimination(order, True)
	
	# Debug
	if debugFlag == True:
		jointTable.printFC()

	# Find the entry with max value
	maxIdx = -1
	maxVal = 0
	maxAss = []
	for i in range(len(jointTable.factTable)):
		# Get the assignment for this entry
		assignment = getVarAssignmentFromFactorIndex(jointTable, i)
		isTheEntry = True
		if len(mn[mn_evidences]) != 0:
			# Only care about the entry satisfying the evidences
			for e in mn[mn_evidences]:
				# Hunt for the index of this evidence var within the factor
				eIdx = jointTable.indexOfVar.index(e[0])
				if assignment[eIdx] != e[1]:
					isTheEntry = False
					break
		if isTheEntry == True:
			# Find the max value
			if jointTable.factTable[i] > maxVal:
				maxVal = jointTable.factTable[i]
				maxAss = assignment
				maxIdx = i

	# Debug
	if debugFlag == True:
		print('maxIdx: ', maxIdx)
		print('maxVal: ', maxVal)
		print('maxAss: ', maxAss)

	# Defensive checking
	if len(maxAss) != len(jointTable.indexOfVar) or len(maxAss) != jointTable.numOfVar or jointTable.numOfVar != len(jointTable.indexOfVar):
		print('Error: internal error for this factor with the assginment list')
		return(None)

	# Construct the (var, val) assignment pair
	mapAss = []
	for j in range(jointTable.numOfVar):
		varVal = []
		isEvid = False
		for e in mn[mn_evidences]:
			if e[0] == jointTable.indexOfVar[j]:
				isEvid = True
				break
		if isEvid == False:
			varVal.append(jointTable.indexOfVar[j])
			varVal.append(maxAss[j])
			mapAss.append(tuple(varVal))
			
	return(mapAss)	
		


def main():
	'''
	Main function
	'''
	# Process parameters
	if len(sys.argv) < 2:
		print('Error: invalid number of parameters')
		return(1)

	filePath = sys.argv[1]
	# Load and parse the file
	loadParseUAIFile(filePath)
	if debugFlag == True:
		printMN()
	#verifyMN()

	# Load and parse the possible evidence file
	if len(sys.argv) == 3:
		if sys.argv[2] == 'bp':
			# Only run BP to save time
			loopyBeliefPropagation()
			computeMarginal4BP()
			return
		elif sys.argv[2] == 've':
			# Only run VE to save time and space
			order = greedyOrdering(0)
			print(mnHeu[0], order)
			pr = variableElimination(order)
			print('ve-heu-0-PR: ', pr)
			return
		else:
			filePath = sys.argv[2]
			loadParseUAIFile(filePath, True)
			if debugFlag == True:
				printMN()

	'''
	UT
	print('_______________________________')
	newFactor = product(mn[mn_factors][1], mn[mn_factors][2], mn[mn_cardinality])
	newFactor.printFC()
	print('_______________________________')
	'''
	# Compute the PR
	pr = computePR4MN()
	print('computePR4MN-PR: ', pr)
	print('_______________________________')
	pr = variableElimination()
	print('ve-PR: ', pr)
	print('_______________________________')
	'''
	UT with ex.uai
	pr = variableElimination([0,1,2])
	print('ve-[0,1,2]-PR: ', pr)
	print('_______________________________')
	pr = variableElimination([0,2,1])
	print('ve-[0,2,1]-PR: ', pr)
	print('_______________________________')
	pr = variableElimination([1,2,0])
	print('ve-[1,2,0]-PR: ', pr)
	print('_______________________________')
	'''
	order = greedyOrdering(0)
	print(mnHeu[0], order)
	pr = variableElimination(order)
	print('ve-heu-0-PR: ', pr)
	print('_______________________________')
	order = greedyOrdering(1)
	print(mnHeu[1], order)
	pr = variableElimination(order)
	print('ve-heu-1-PR: ', pr)
	print('_______________________________')
	order = greedyOrdering(2)
	print(mnHeu[2], order)
	pr = variableElimination(order)
	print('ve-heu-2-PR: ', pr)
	print('_______________________________')
	order = greedyOrdering(3)
	print(mnHeu[3], order)
	pr = variableElimination(order)
	print('ve-heu-3-PR: ', pr)
	print('_______________________________')
	pr = computePROB4MNWithEvid()
	print('prob-of-evidence: ', pr)
	print('_______________________________')
	pr = computeMAP4MN()
	print('MAP: ', pr)
	print('_______________________________')

	# BP
	loopyBeliefPropagation()
	computeMarginal4BP()


if __name__ == '__main__':
    main()

