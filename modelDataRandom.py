import torch
import torch.nn as nn
import sequencesBuilder
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

###### Initialize ######
batchSize = 64
epochs = 100

class LSTMModel(nn.Module):
    def __init__(self, inputSize = 4, hiddenLayerSize = 100, outputSize = 1):
        super(LSTMModel, self).__init__()
        self.hiddenLayerSize = hiddenLayerSize
        self.lstmModel = nn.LSTM(inputSize, hiddenLayerSize, batch_first=True)
        self.linear = nn.Linear(hiddenLayerSize, outputSize)
        self.sigmoid = nn.Sigmoid() #probability of the input being 1 (stock up)

    def forward(self, inputSequence):
         # Assuming inputSequence shape: [sequence_length, batch_size, features]
        lstmOutput, _ = self.lstmModel(inputSequence)
        # print(lstmOutput.shape)
        
        
        final_outputs = lstmOutput[:,-1,:]
        # print(final_outputs.shape)

        
        predictions = self.linear(final_outputs)
        # print(predictions)

        # Sigmoid = Probabilities
        predictions = self.sigmoid(predictions)
        # print(predictions)

        
        return predictions

###### Create Tensor Dict ######
tickerTensors = dict()
for ticker, (sequences, results) in sequencesBuilder.tickerSeq.items():
    seqTensor = torch.FloatTensor(sequences)
    total_elements = seqTensor.numel()  # Total number of elements in the tensor
    sequence_length = 10  # Number of timesteps per sequence
    features_per_timestep = 4  # Number of features per timestep
    # Calculate the number of sequences based on the total number of elements
    num_sequences = total_elements // (sequence_length * features_per_timestep)
    seqTensor = seqTensor.view(num_sequences, sequence_length, features_per_timestep)

    targTensor = torch.tensor(results, dtype=torch.float).view(-1, 1)
    tickerTensors[ticker] = (seqTensor, targTensor)

###### Train Function ######
def trainModel(model, epochs, optimizer, lossFunction, trainLoader):
    model.train()
    for epoch in range(epochs):
        for seqBatch, targBatch in trainLoader:
            seqBatch = torch.nan_to_num(seqBatch, nan=0.0)
            optimizer.zero_grad()
            yPred = model(seqBatch)
            # print("yPred:", yPred)
            # print("targBatch:", targBatch)
            loss = lossFunction(yPred, targBatch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            if epoch % 45 == 0:  # Print the loss every 45 epochs
                print(f'Epoch {epoch} loss: {loss.item()}')

###### Validate Function ######
def validateModel(model, valLoader, lossFunction):
    model.eval()
    totalLoss = 0
    correctPred = 0
    totalPred = 0
    with torch.no_grad():
        for seqBatch, targBatch in valLoader:
            predictions = model(seqBatch)
            loss = lossFunction(predictions, targBatch)
            totalLoss += loss.item()

            predictedClasses = predictions.round()  # Assuming sigmoid output
            correctPred += (predictedClasses == targBatch).sum().item()
            totalPred += targBatch.size(0)
    avgLoss = totalLoss / len(valLoader)
    accuracy = correctPred /totalPred
    print(f'Validation Loss: {avgLoss}, Accuracy: {accuracy}')
    return avgLoss, accuracy


###### Iterate ######

modelPath = 'model_state_dict.pth'
bestValLoss = float('inf')
bestValAccuracy = 0.0  
model = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
lossFunction = nn.BCELoss()
for ticker, (seqTensor, targTensor) in tickerTensors.items():
    print(ticker)
    # Tensors --> numPy
    sequences_np = seqTensor.numpy()
    targets_np = targTensor.numpy()

    # Split
    seq_train_np, seq_val_np, targ_train_np, targ_val_np = train_test_split(
        sequences_np, targets_np, test_size=0.2, random_state=42)
    
    # numPy --> Tensors
    seq_train, targ_train = torch.tensor(seq_train_np), torch.tensor(targ_train_np)
    seq_val, targ_val = torch.tensor(seq_val_np), torch.tensor(targ_val_np)

    # Tensors
    train_dataset = TensorDataset(seq_train, targ_train)
    val_dataset = TensorDataset(seq_val, targ_val)

    # Dataloader
    trainLoader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)

    

    trainModel(model, epochs, optimizer, lossFunction, trainLoader)
    loss, accuracy = validateModel(model, valLoader, lossFunction)
    if loss <= bestValLoss:
        bestValLoss = loss
        torch.save(model.state_dict(), 'model_best_val_loss.pth')
        print(f"Model saved  with best validation loss: {bestValLoss}")

    if accuracy >= bestValAccuracy:
        bestValAccuracy = accuracy
        torch.save(model.state_dict(), 'model_best_val_accuracy.pth')
        print(f"Model saved with best validation accuracy: {bestValAccuracy}")

torch.save(model.state_dict(), 'model_final.pth')
print(f"Final model saved")
