#!/usr/bin/env python3

"""
Author: Zack Oldham
NLP Project 2
Textual Entailment: Determine if a premise entails a hypothesis using a neural network
03/30/2020
"""


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils import data
from sklearn.metrics import classification_report
import xml.etree.ElementTree as xml



INT_ENC = {}  # dict of integer word encodings
ENC_COUNT = 1  # the next available unique encoding for a word
BATCH = 100



"""
plan for neural net:
	Embedding Layer: Fairly shallow linear layer with sigmoid activation (get rid of as many zeroes as possible) (p and h each go thru separate)
	LSTM Layer: An LSTM layer to get temporal representations of the vectors (p and h each go thru separate)
	Fully Connected Layer: Series of linear layers that converges to single output 
	(0 or 1 to represent not entails or entails) (concat p and h output from LSTM layer as input to this layer)
"""

class TextEntailModel(nn.Module):
	def __init__(self, input_size):
		super().__init__()
		self.input_size = input_size
		self.embed_size = 64
		self.lstm_hidden = 32
		self.embed = nn.Linear(self.input_size, self.embed_size)
		self.LSTM = nn.LSTM(input_size=self.embed_size, hidden_size=self.lstm_hidden, num_layers=2, bidirectional=True)
		self.fc1 = nn.Linear(self.lstm_hidden*4, 64)
		self.fc2 = nn.Linear(64, 32)
		self.fc3 = nn.Linear(32, 16)
		self.fc4 = nn.Linear(16, 2)
		self.output = nn.Softmax(dim=1)
		self.sigm = nn.Sigmoid()




	# send a batch of inputs through the system (embed layer -> LSTM layer -> fully connected layer -> output)
	def forward(self, p, h, current_batch, train=False):
		# embed the inputs to condense tensors
		p_em = self.sigm(self.embed(p))
		h_em = self.sigm(self.embed(h))

		# convert tensors into shape compatible with LSTM
		p_em = p_em.view(1, current_batch, self.embed_size)
		h_em = h_em.view(1, current_batch, self.embed_size)


		# obtain temporal sequences for each tensor from LSTM
		p_t = self.LSTM(p_em)[0].view(current_batch, self.lstm_hidden*2)
		h_t = self.LSTM(h_em)[0].view(current_batch, self.lstm_hidden*2)

		# concatenate results into one tensor
		ph_t = torch.cat((p_t, h_t), dim=1)

		# pass through fully connected layers to obtain output (in the form of probability distribution)
		out = self.sigm(self.fc1(ph_t))
		out = self.sigm(self.fc2(out))
		out = self.sigm(self.fc3(out))
		out = self.output(self.fc4(out))

		if not train:
			out = self.from_prob_dist(out)

		return out


	# convert a set of labels into probability distributions
	def to_prob_dist(self, labels):
		y_dist = []
		for i in range(len(labels)):
			if labels[i] == 0.0:
				y_dist.append([1.0, 0.0])
			else:
				y_dist.append([0.0, 1.0])

		return torch.tensor(y_dist)


	# convert a set of probability distributions into integer class labels
	def from_prob_dist(self, y_dist):
		y = np.array([]) 
		for row in y_dist:
			y.append(torch.argmax(row)[0])

		return y



	# define how to train the neural net
	def fit(self, P, H, labels, current_batch=BATCH, epoch=20):
		loss_fn = nn.L1Loss()
		optimizer = optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
		for i in range(epoch):
			self.train()  # signify that we are in training mode
			for j in range(len(P)):  # We expect data to be a batch of training examples
				optimizer.zero_grad()  # zero the gradient for this fresh data set
				Y_pred = self(P, H, current_batch, train=True)
				Y_true = self.to_prob_dist(labels)
				loss = loss_fn(Y_pred, Y_true)  # how wrong were we? (the smaller this gets the better)
				loss.backward()  # perform backward propagation to adjust weights
				optimizer.step()  # take the next gradient descent 'step' (hopefully nearer to the bottom of the 'canyon')








# import all of the premise/hypothesis/class tuples from the xml file
def import_data(filename):
	root = xml.parse(filename).getroot()
	p = []
	h = []
	c = []

	for child in root:
		p.append(child.find('t').text.split(' '))
		h.append(child.find('h').text.split(' '))
		c.append(0) if child.get('value') == 'FALSE' else c.append(1)

	return p, h, c



# convert a sentence to its integer encoding (each word has its own unqiue encoding)
def encode_sentence(sentence):
	global INT_ENC, ENC_COUNT

	sent_enc = []

	for word in sentence:
		if word not in INT_ENC:
			INT_ENC[word] = ENC_COUNT
			ENC_COUNT += 1
		
		sent_enc.append(INT_ENC[word])

	return sent_enc



def pad_encodings(prems, hyps, max_len):
	for i in range(len(prems)):
		if len(prems[i]) < max_len:
			zeroes = max_len - len(prems[i])
			for j in range(zeroes):
				prems[i].append(0)

		if len(hyps[i]) < max_len:
			zeroes = max_len - len(hyps[i])
			for j in range(zeroes):
				hyps[i].append(0)


	return prems, hyps




# get the data in the given filename and convert to TensorDatasets
def get_datasets(filename):
	p_raw, h_raw, Y_true = import_data(filename)

	enc_p = []
	enc_h = []

	max_len = 0

	for p,h in zip(p_raw, h_raw):
		enc_p.append(encode_sentence(p))
		if len(p) > max_len:
			max_len = len(p)

		enc_h.append(encode_sentence(h))
		if len(h) > max_len:
			max_len = len(h)


	return enc_p, enc_h, Y_true, max_len




def to_dataset(arr):
	tens = torch.tensor(arr).float()
	dataset = data.TensorDataset(tens)
	return dataset


def main():
	# 1. prepare dataset
	# 2. prepare inputs for training/testing (form batches)
	# 3. Define Model
	# 4. Train/Test Model --> Display accuracy report

	P_train, H_train, Y_train, trn_len = get_datasets('train.xml')
	P_test, H_test, Y_test, tst_len = get_datasets('test.xml')

	if trn_len > tst_len:
		max_len = trn_len
	else:
		max_len = tst_len


	P_train, H_train = pad_encodings(P_train, H_train, max_len)
	P_test, H_test = pad_encodings(P_test, H_test, max_len)


	P_train = to_dataset(P_train)
	H_train = to_dataset(H_train)
	Y_train = to_dataset(Y_train)
	P_test = to_dataset(P_test)
	H_test = to_dataset(H_test)
	Yt_set = to_dataset(Y_test)


	P_rand_sampler = data.RandomSampler(P_train)
	P_train_loader = data.DataLoader(P_train, batch_size=BATCH, sampler=P_rand_sampler)

	H_rand_sampler = data.RandomSampler(H_train)
	H_train_loader = data.DataLoader(H_train, batch_size=BATCH, sampler=H_rand_sampler)

	Y_rand_sampler = data.RandomSampler(Y_train)
	Y_train_loader = data.DataLoader(Y_train, batch_size=BATCH, sampler=Y_rand_sampler)


	entail_model = TextEntailModel(max_len)

	for p_trn, h_trn, y_trn in zip(P_train_loader, H_train_loader, Y_train_loader):
		if p_trn[0].shape[0] < BATCH:
			entail_model.fit(p_trn[0], h_trn[0], y_trn[0], current_batch=p_trn[0].shape[0])
		else:
			entail_model.fit(p_trn[0], h_trn[0], y_trn[0])


	P_seq_sampler = data.SequentialSampler(P_test)
	P_test_loader = data.DataLoader(P_test, batch_size=BATCH, sampler=P_seq_sampler)

	H_seq_sampler = data.SequentialSampler(H_test)
	H_test_loader = data.DataLoader(H_test, batch_size=BATCH, sampler=H_seq_sampler)


	Y_pred = np.array([])

	for p_tst, h_tst, y_tst in zip(P_test_loader, H_test_loader):
		if p_tst[0].shape[0] < BATCH:
			y_pred = entail_model(p_tst, h_tst, p_tst[0].shape[0])
		else:
			y_pred = entail_model(p_tst, h_tst)

		Y_pred = np.concatenate(Y_pred, y_pred, axis=None)




	Y_test = np.array(Y_test)

	classification_report(Y_test, Y_pred)

	
		



if __name__ == '__main__':
	main()




