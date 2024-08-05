import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None
        super().__init__()

    def forward(self, prediction_tensor, label_tensor):
        # Adding a small epsilon value for numerical stability
        epsilon = np.finfo(float).eps
        # Computing the cross-entropy loss
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        #B = label_tensor.shape[0]
        ce_loss = -np.sum(label_tensor * np.log(prediction_tensor + epsilon)) 
        return ce_loss


    def backward(self, label_tensor):
        epsilon = np.finfo(float).eps
        prediction_tensor = np.clip(self.prediction_tensor, epsilon, 1 - epsilon)
        # Compute gradient of the loss with respect to predictions
        gradient_tensor = - (label_tensor / prediction_tensor) #/ prediction_tensor.shape[0]
        return gradient_tensor
