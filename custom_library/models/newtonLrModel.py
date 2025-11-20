import copy
import numpy as np

from ..core import BaseLrModelImplementHessain
from ..utils import metrics

class NewtonLrModel(BaseLrModelImplementHessain):
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