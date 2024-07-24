import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from utils import bag_of_words, tokenize, stem
from model import TranNet
import torch.nn as nn
import os

class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

class TrainData:
    def __init__(self, data_file_path='intent.json', save_file_path='model/intent_model.pth'):
        self.file_path = data_file_path
        self.save_file_path = save_file_path
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(self.save_file_path), exist_ok=True)
        
        with open(self.file_path, 'r') as f:
            intents = json.load(f)
        
        self.all_words = []
        self.tags = []
        xy = []

        for intent in range(len(intents)):
            tag = intents[intent]['intent']
            self.tags.append(tag)
            for pattern in intents[intent]['patterns']:
                w = tokenize(pattern)
                self.all_words.extend(w)
                xy.append((w, tag))

        ignore_words = ['?', '.', '!']
        self.all_words = [stem(w) for w in self.all_words if w not in ignore_words]
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))

        print(len(xy), "patterns")
        print(len(self.tags), "tags:", self.tags)
        print(len(self.all_words), "unique stemmed words:", self.all_words)

        X_train = []
        y_train = []
        for (pattern_sentence, tag) in xy:
            bag = bag_of_words(pattern_sentence, self.all_words)
            X_train.append(bag)
            label = self.tags.index(tag)
            y_train.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.batch_size = 16
        self.learning_rate = 0.001
        self.input_size = len(X_train[0])
        self.hidden_size = 16
        self.output_size = len(self.tags)

        dataset = ChatDataset(X_train, y_train)
        self.train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = TranNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def trainModel(self, num_epochs=1000):
        for epoch in range(num_epochs):
            for (words, labels) in self.train_loader:
                words = words.to(self.device)
                labels = labels.to(dtype=torch.long).to(self.device)
                
                outputs = self.model(words)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')

        print(f'final loss: {loss.item():.8f}')

        data = {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "all_words": self.all_words,
            "tags": self.tags
        }

        print(f'Saving model to {self.save_file_path}')
        torch.save(data, self.save_file_path)
        print(f'Training complete. File saved to {self.save_file_path}')
        
model = TrainData()
model.trainModel()
