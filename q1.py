from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from collections import Counter
import math
import json 


simScores = {}
docs = []
docsList = []
tfidf = {}
docsString = ""
corpus = {}
idf = {}	
queryDict = {}

def processDocuments():
	file = open("data.txt", "r");
	docsString = file.read()
	docs = docsString.split("\n")
	for doc in docs:
		tokenizer = RegexpTokenizer(r'\w+')
		wordList = tokenizer.tokenize(doc)
		docList = []
		for word in wordList:
			lemmaWord = lemmatize(word)
			lemmaWord = lemmaWord.lower()
			docList.append(lemmaWord)
		docsList.append(docList)	
	# print(docsList[0])	


def lemmatize(word):
	lemmatizer = WordNetLemmatizer()
	wordLemma = lemmatizer.lemmatize(word)
	return wordLemma

docWordFreq  = {}

def getCorpusWords():
	wordList = []
	for i, doc in enumerate(docsList):
		docWordFreq[i] = dict(Counter(doc))
		for word in doc:
			wordList.append(word)
	# print(corpus)		
	return dict(Counter(wordList))


def generateTfidfMatrix():
	tfidf = {}
	for word in corpus:
		# word = corpus[key]
		tfidf[word] = {}
	
		nt = idf[word]	
		for i, doc in enumerate(docsList):	
			if(word in doc): 
				idfValue = math.log(float(1000)/float((1+nt)), 2)
				tfidfValue = getTfValue(word, i)*idfValue
				# tfidfVector.append(tf)
				tfidf[word][i] = tfidfValue
			
	return tfidf		


def getTfValue(word, i):
	# print(docWordFreq[i])
	tf = docWordFreq[i][word]
	return (1+math.log((1+tf), 2))

def getIdfValues():
	idf = {}
	for word in corpus:
		idf[word] = 0
		for doc in docsList:
			if(word in doc):
				idf[word]+=1

	return idf			

def getQueries():
	file = open("test.jsonl", "r")
	jsonq = file.read()
	# print(jsonq)
	jsonList = jsonq.split("\n")
	# queryDict = {}
	# print(jsonList[0])
	for i, query in enumerate(jsonList):
		queryDict[i] = json.loads(query)
		# print("--------------------------------", i)
	# print(queryDict)	


def getQueryvector(queryList):
	queryVector = {}
	occurList = []
	# print(idf)
	for word in queryList:
		if word not in occurList:
			count = queryList.count(word)
			tf = 1 + math.log(1+count, 2)
			if(word in idf):
				nt = idf[word]
			else:
				nt = 0
			# nt = idf[word]
			idfValue = math.log(float(1000)/float((1+nt)), 2)
			tfidf = tf*idfValue
			queryVector[word] = tfidf
			occurList.append(word)
	return queryVector		

def cosineSimilarity(queryList, queryVector, doc):
	# lq = len(queryVector)
	# ld = len(docVector)
	dotProduct = 0
	# if(lq < ld):
	# 	for x in range(0, lq):
	# 		dotProduct+=queryVector[x]*docVector[x]
	modQ = modD = 0

	for word in queryList:
		docVal = 0
		if (word in tfidf and doc in tfidf[word]):
			docVal = tfidf[word][doc]
		dotProduct+=queryVector[word]*docVal	
		modQ+=queryVector[word]*queryVector[word]
		modD+=docVal*docVal

	# for val in queryVector:
	# 	modQ+=val*val
	# modQ = math.sqrt(modQ)
	# for val in docVector:
	# 	modD+=val*val
	modQ = math.sqrt(modQ)
	modD = math.sqrt(modD)

	if(modD==0 or modQ==0):
		return 0
	similarity = (float(dotProduct)/(float(modQ*modD)))

	return similarity	

def getMaxScore(maxList, correctAnswer):
	maxList.sort(key = lambda x: x[1])
	maxList = maxList[::-1]
	print(maxList)
	score = 0
	if(maxList[0][0]==correctAnswer):
		score = 1
		for answer in maxList[1:]:
			if(answer[0]==correctAnswer):
				score+=1
	# print(score)			
	return score			


processDocuments()
corpus = getCorpusWords()	
# print(corpus)
# print(docWordFreq[0])
# print(tfidf)
idf = getIdfValues()
tfidf = generateTfidfMatrix()
getQueries()
# print(tfidf)
getQueries()	
# print(queryDict[1])

# print(tfidf)

# print(queryDict[0])
# print(docWordFreq)
accuracy = 0
correct = 0
finalScore = 0
for key in queryDict:
	qSimilarity = []
	maxSimilarity = 0
	question = queryDict[key]
	# print(question)
	answers = question['question']['choices']
	query =  question['question']['stem']
	# print(query)
	maxList = []
	simScores[key]  = {}
	simScores[key]['id'] = queryDict[key]['id']
	simScores[key]['choices'] = []
	for answer in answers:
		maxSimilarity = 0
		queryList = []
		queryAnswer = query + " " + answer['text']
		tokenizer = RegexpTokenizer(r'\w+')
		queryWords = tokenizer.tokenize(queryAnswer)

		for word in queryWords:
			queryList.append(lemmatize(word).lower())

		queryVector = getQueryvector(queryList)
		for i, doc in enumerate(docsList):
			cosineSim = cosineSimilarity(queryList, queryVector, i)
			if(cosineSim > maxSimilarity):
				maxSimilarity = cosineSim
				maxAnswer = answer['label']
				maxDoc = i
		# qSimilarity.append()
		# query = question + " " + 
		maxList.append((maxAnswer,maxSimilarity))
		dikt = {}
		dikt['label'] = maxAnswer
		dikt['similarity'] = maxSimilarity
		dikt['doc'] = maxDoc
		simScores[key]['choices'].append(dikt)
	score = getMaxScore(maxList, queryDict[key]['answerKey'])	
	# print(maxSimilarity, maxAnswer)
	if(score!=0):
		finalScore+=(float(1/score))	

# print(simScores)
print(finalScore/500)
