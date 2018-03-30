#!/usr/bin/env python2.7

from collections import Counter
from tqdm import tqdm
import numpy as np
from scipy import sparse

corpus_file = '/Users/calvin-is-seksy/Desktop/myProjects/CS291A/data/text8'
vocab_file = '/Users/calvin-is-seksy/Desktop/myProjects/CS291A/data/vocab.txt'
output_file = './vector.txt'
LR = 1e-2
epoch = 20
CWS = 4
numNS = 5 
vector_dimension = 300 
a = .75
xMax = 100

def lines(infile):
    with open(infile, 'r') as fp:
        for line in fp:
            yield line

def run(corpus_file, vocab_file, output_file):

	# read data from input
	allTargetWords = []
	for line in lines(vocab_file):
		temp1 = line.strip() 
		temp2 = temp1.split()[0]
		word = temp2.lower()
		allTargetWords.append(word)

	myCounter = Counter() 
	for line in lines(corpus_file):
		temp1 = line.strip()
		words = line.split()
		myCounter.update(words)

	vocab = {}
	for index, word in enumerate(allTargetWords):
		vocab[word] = (index, myCounter[word])

	vocabSize = len(vocab)

	coocurence_matrix = [] 
	coocurence_matrix_sparse = sparse.lil_matrix((vocabSize, vocabSize), dtype=np.float64)
	index_to_word = dict((index, word) for word, (index, _) in vocab.iteritems())

	for line_index, l in enumerate(lines(corpus_file)): 
		words  = l.strip().split()
        tokenIDs = [vocab.get(w, (None, 0))[0] for w in words]

        for centerIndex, centerID in enumerate(tokenIDs):
        	contextIDs = tokenIDs[max(centerIndex-CWS, 0) : centerIndex]
        	contextLength = len(contextIDs)
        	for leftIndex, leftID in enumerate(contextIDs):
        		if centerID == None or leftID == None: continue 
        		d = CWS - leftIndex
        		coocurence_matrix_sparse[centerID, leftID] += 1.0/float(d)
        		coocurence_matrix_sparse[leftID, centerID] += 1.0/float(d)

	for centerToken, (row, dataRow) in enumerate(zip(coocurence_matrix_sparse.rows, coocurence_matrix_sparse.data)):
		for contextToken, data in zip(row, dataRow):
			coocurence_matrix.append((centerToken, contextToken, data))

	# your training algorithm
	embedding_word = np.random.uniform(-.5, .5, size=(vocabSize,vector_dimension))
	bias_word = np.random.uniform(-.5, .5, size=vocabSize)
	embedding_context = np.random.uniform(-.5, .5, size=(vocabSize,vector_dimension))
	bias_context = np.random.uniform(-.5, .5, size=vocabSize)
	gradient_squared_embedding_word = np.ones((vocabSize, vector_dimension))
	gradient_squared_bias_word = np.ones(vocabSize)
	gradient_squared_embedding_context =np.ones((vocabSize, vector_dimension))
	gradient_squared_bias_context = np.ones(vocabSize)

	for epo in range(epoch): 
		loss = 0 
		np.random.shuffle(coocurence_matrix)

		for MT, CT, c in tqdm(coocurence_matrix): 
			w = (c/xMax)**a if c < xMax else 1 

			temp_EW = embedding_word[MT]
			temp_BW = bias_word[MT]
			temp_EC = embedding_context[CT]
			temp_BC = bias_context[CT]
			temp_GSEW = gradient_squared_embedding_word[MT]
			temp_GSBW = gradient_squared_bias_word[MT]
			temp_GSEC = gradient_squared_embedding_context[CT]
			temp_GSBC = gradient_squared_bias_context[CT]

			in_cost = np.dot(temp_EW, temp_EC) + temp_BW + temp_BC - np.log(c) 
			loss += w * np.square(in_cost) / 2

			temp_GBC = temp_GBW = w * in_cost
			temp_GEW = temp_GBW * temp_EC
			temp_GEC = temp_GBC * temp_EW
			
			temp_EW -= LR*temp_GEW / (temp_GSEW**.5)
			temp_BW -= LR*temp_GBW / (temp_GSBW**.5)
			temp_EC -= LR*temp_GEC / (temp_GSEC**.5)
			temp_BC -= LR*temp_GBC / (temp_GSBC**.5)

			temp_GSEW += temp_GEW**2
			temp_GSBW += temp_GBW**2 
			temp_GSEC += temp_GEC**2 
			temp_GSBC += temp_GBC**2 

		print 'Epoch: %d \t Loss: %.5f' % (epo+1, loss/len(coocurence_matrix)) 
		embedding = ((embedding_word + embedding_context) / 2) 
		embedding = embedding / np.linalg.norm(embedding, axis=-1).reshape(-1, 1)

	# your prediction code
	with open('./vectors.txt', 'w') as out: 
		for element in allTargetWords: 
			vec = ['%.6f' % tempWeight for tempWeight in embedding[vocab[element][0]]]
			myLine = element + " " + " ".join(vec) + '\n'
			out.write(myLine)

if __name__ == '__main__':
	run(corpus_file, vocab_file, output_file)

























