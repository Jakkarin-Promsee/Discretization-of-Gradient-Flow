import copy
import numpy as np

from ..core import BaseLrModel
from ..layers import DenseLayer, ActivationLayer
from ..utils import hessainUtils, metrics

class NewtonLrModel(BaseLrModel):
    def flatten_params(self, layers):
        flat_list = []
        shapes = []

        for layer in layers:
            if(isinstance(layer, ActivationLayer)):
                continue

            if hasattr(layer, "W"):
                shapes.append(("W", layer.W.shape))
                flat_list.append(layer.W.ravel())

            if hasattr(layer, "b"):
                shapes.append(("b", layer.b.shape))
                flat_list.append(layer.b.ravel())

        return np.concatenate(flat_list), shapes
    
    def unflatten_params(self, layers, theta_vec, shapes):
        i=0
        idx = 0
        for layer in layers:
            if(isinstance(layer, ActivationLayer)):
                continue
            
            if hasattr(layer, "W"):
                name, shape = shapes[i]
                size = np.prod(shape)
                layer.W = theta_vec[idx:idx+size].reshape(shape)
                idx += size
                i += 1

            if hasattr(layer, "b"):
                name, shape = shapes[i]
                size = np.prod(shape)
                layer.b = theta_vec[idx:idx+size].reshape(shape)
                idx += size
                i += 1

    def compute_gradient(self, layers, X, y_true):
        # Forward pass
        a = np.array(X)
        zal = [a] # to keep both z or a

        for layer in layers:
            # If now we're on dense layer
            if(isinstance(layer, DenseLayer)):
                if(a.shape[1] != layer.input_dim):
                    raise ValueError("Input dimension mismatch: expected " + 
                                    f"{layer.input_dim}, got {a.shape[1]}")
                a = np.dot(a, layer.W) + layer.b

            # If now we're on activation layer
            if(isinstance(layer, ActivationLayer)):
                a = layer.forward(a)

            # keep forward data
            zal.append(a)

        # Get zi
        y_pred = a

        # dL/dz (MSE) = (2/N)*(zi-yi)
        dL_dz = (2/X.shape[0]) * (y_pred - y_true)

        # Set iterator
        i = len(zal)-1 

        gradient = []

        # Compute gradients and Update weights
        for layer in reversed(list(layers)): 
            # cause i=max() have done calculate at first dL/dz
            i-=1

            if(isinstance(layer, DenseLayer)):           
                # Compute Gradient     
                dL_dW = np.dot(zal[i].T, dL_dz)
                dL_db = np.sum(dL_dz, axis=0, keepdims=True)

                # Store Gradient
                gradient.insert(0,dL_dW)
                gradient.insert(0, dL_db)
                
                # Compute next delta
                dL_dz = np.dot(dL_dz, layer.W.T)

            if(isinstance(layer, ActivationLayer)):
                dL_dz = layer.backward(zal[i]) * dL_dz

        return np.concatenate([g.ravel() for g in gradient])

    
    def compute_hessian(self, layers, X, y, epsilon=1e-2):
        theta, shapes = self.flatten_params(layers)
        
        N = theta.size
        H = np.zeros((N, N))

        for j in range(N):
            e = np.zeros(N)
            e[j] = 1

            # theta + eps
            theta_pos = theta + epsilon * e
            layers_pos = copy.deepcopy(layers)
            self.unflatten_params(layers_pos, theta_pos, shapes)
            g_pos = self.compute_gradient(layers_pos, X, y)

            # theta - eps
            theta_neg = theta - epsilon * e
            layers_neg = copy.deepcopy(layers)
            self.unflatten_params(layers_neg, theta_neg, shapes)
            g_neg = self.compute_gradient(layers_neg, X, y)

            # second derivative column j
            H[:, j] = (g_pos - g_neg) / (2 * epsilon)
        return H

    def fit(self, X_train, y_train, X_eval, y_eval, epochs=100, batch_size=32):
        """train models"""
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                 # Fetch the batch
                y_true_batch = y_train[batch_start:batch_end]
                X_batch = X_train[batch_start:batch_end]

                # Newton method
                theta, shapes = self.flatten_params(self.layers)
                H = self.compute_hessian(self.layers, X_batch, y_true_batch)
                g = self.compute_gradient(self.layers, X_batch, y_true_batch)

                # Using damping to prevent det(H) = 0
                damping_factor = 1e-4 
                H_damped = H + damping_factor * np.eye(H.shape[0])

                # H * theta = g
                # theta = solve(H,g)
                delta = -np.linalg.solve(H_damped, g)   

                print(delta)

                # Compute new theta
                theta_new = theta + delta

                # Update Weight
                self.unflatten_params(self.layers, theta_new, shapes)

            # Predict
            pred_train = self.predict(X_train)
            pred_eval = self.predict(X_eval)

            # Evaluate
            train_acc = metrics.mae(y_train, pred_train)
            val_acc = metrics.mae(y_eval, pred_eval)
            
            # Save
            self.history.save(train_acc, val_acc)
            self.history.save_predict(X_eval, pred_eval, y_eval)
            self.history.save_model(self.layers, val_acc)

            print(f"Epoch {epoch+1}/{epochs} [", end="")

            progress_bar_length = 25
            progress = int((epoch/epochs)*progress_bar_length)
            for i in range(progress_bar_length):
                if(i<=progress):
                    print("=", end="")
                else:
                    print(".", end="")
            print("]")
            print(f"loss: {train_acc:.4f}, val_loss: {val_acc:.4f}")
            print("") 
        print(f"best-loss: {self.history.get_best_loss():.4f}")
        return self.history