import torch
import torch.nn as neuralNetwork
import torch.nn.functional as functional
import os
import torch.optim as optim

class Linear_QNet(neuralNetwork.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # initialize neural network
        self.linearIn = neuralNetwork.Linear(input_size, hidden_size).cuda()
        self.linearOut = neuralNetwork.Linear(hidden_size, output_size).cuda()
    
    #specifies how input data is propagated through the layers of the network to produce an output
    def forward(self,x):
        #Applies relu to the output of the input layer to introduce non linearity
        x = functional.relu(self.linearIn(x)) #relu = rectified linear unit

        # passes the output of the ReLU activation function through the output layer
        # This layer produces the final output of the model
        x = self.linearOut(x)
        return x
    
    # saves the model's trained weights to a file
    def save(self, file_name='model_name.pth'):
        model_folder_path = 'Path'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), "save.pth")

class QTrainer:
    def __init__(self, model, lr, gamma):
        
        #learning rate of the algorithm, which controls how much the weights of 
        #the model are updated in response to each batch of training examples
        # higher eaquals faster learning but less stable/accurate results
        # lower = slower leaning but more accurate and stable seults
        self.lr = lr

        #discount factor used in the Q-learning algorithm
        # higher equals prioritizes more long term rewards
        # lower equals prioritizses more immediate rewards
        self.gamma = gamma

        #model/neural network
        self.model = model

        #used to update the weights of a neural network during training
        #model.paramerters is the list of the parameters which are the weights and biases of the models layers
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        
        #(MSE) loss function is commonly used as a measure of how well a neural network model is performing on a given task
        self.criterion = neuralNetwork.MSELoss()
        for i in self.model.parameters():
            print(i.is_cuda)

    def train_step(self, state, action, reward, next_state, done):
        #cuda is used for faster computation. It uses the GPU

        #convert raw input data into tensors which are used by pytorch
        #for building and training neural networks
        state = torch.tensor(state, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.long).cuda()
        reward = torch.tensor(reward, dtype=torch.float).cuda()

        #add an extra batch dimension to the tensor, which is often necessary for processing the data in a neural network
        #when the data is a vector(size of 1)
        if(len(state.shape) == 1):
            state = torch.unsqueeze(state, 0).cuda()
            next_state = torch.unsqueeze(next_state, 0).cuda()
            action = torch.unsqueeze(action, 0).cuda()
            reward = torch. unsqueeze(reward, 0).cuda()
            done = (done, )
        
        # output of the neural network
        predictedValue = self.model(state).cuda()

        # will be modified later
        targetValue = predictedValue.clone().cuda()

        for idx in range(len(done)):
            #  If done[idx] is True, then the Q value is simply set to the reward
            Q_new = reward[idx]
            if not done[idx]:
                # If done[idx] is False, then the Q value is set to the sum of the reward reward[idx] and the discounted maximum predicted Q value for the next state self.gamma * torch.max(self.model(next_state[idx])).cuda()
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).cuda()
            # Q value for the action with the highest predicted value (as determined by torch.argmax(action).item()) is set to the new Q value computed above
            targetValue[idx][torch.argmax(action).item()] = Q_new
        
        # sets all gradients of the model parameters to zero to avoid accumulation of gradients from previous iterations.
        self.optimizer.zero_grad()

        #computes the loss between the predicted values and the target values.
        loss = self.criterion(targetValue, predictedValue)

        #computes the gradients of the loss with respect to the model parameters using backpropagation.
        loss.backward()
        
        #updates the model parameters using the computed gradients and the optimizer's update rule.
        #The optimizer uses the gradients to adjust the weights in a direction that minimizes the loss.
        self.optimizer.step()