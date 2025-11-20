import copy
import numpy as np

from ..core import BaseLrModelImplementHessain
from ..utils import metrics

class OptimizeNewtonLrModel(BaseLrModelImplementHessain):
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
                damping_factor = 1e-3 
                H_damped = H + damping_factor * np.eye(H.shape[0])

                # Calculate Newton step direction (delta)
                try:
                    # Solve: (H + lambda*I) * delta = -g
                    delta = -np.linalg.solve(H_damped, g)
                except np.linalg.LinAlgError:
                    # Fallback if the damped matrix is still singular (rare)
                    print("Warning: Damped Hessian is singular. Skipping step.")
                    continue  

                # --- Backtracking Line Search ---
                alpha = 1.0
                c = 1e-4  # Constant for Armijo condition (typically small)
                p = 0.2   # Factor to decrease alpha by (typically 0.1 to 0.5)

                # Get current loss (J_old)
                J_old = self.compute_loss(self.layers, X_batch, y_true_batch)
                
                # Calculate required loss decrease for Armijo condition: g.T @ delta
                g_T_delta = np.dot(g.T, delta)

                # Create a deep copy of layers for testing steps
                test_layers = copy.deepcopy(self.layers)

                # Backtracking loop
                while alpha > 1e-8: # Prevent infinite loop, use small threshold
                    # 3. Compute new parameters for current alpha
                    theta_new = theta + alpha * delta
                    
                    # Update test layers with theta_new
                    self.unflatten_params(test_layers, theta_new, shapes)
                    
                    # Compute new loss (J_new)
                    J_new = self.compute_loss(test_layers, X_batch, y_true_batch)
                    
                    # Armijo Condition: J_new <= J_old + c * alpha * g.T @ delta
                    # (Since delta is the descent direction, g.T @ delta is < 0)
                    if J_new <= J_old + c * alpha * g_T_delta:
                        # Found sufficient decrease! Break loop and accept this alpha
                        break
                    
                    # Decrease alpha
                    alpha *= p
                
                # If line search failed to find an acceptable step (very small alpha)
                if alpha <= 1e-8:
                    print("Warning: Line search failed to find descent step. Skipping step.")
                    continue # Skip update for this batch
                    
                # --- Update Weight (using the accepted theta_new) ---
                # Use the accepted theta_new (computed at the end of the line search)
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
                print(f"Batch {batch_start}/{n_samples}: acc: {train_acc}, val: {val_acc}")

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