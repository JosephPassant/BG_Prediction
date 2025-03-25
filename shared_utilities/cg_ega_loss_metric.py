
import torch
import torch.nn as nn
import numpy as np

class P_EGA_Loss:
    """
    The Point-Error Grid Analysis (P-EGA) estimates the clinical acceptability of glucose predictions
    based on their point accuracy. Implementation matches exactly with metrics.py.
    """
    def __init__(self, y_true, dy_true, y_pred):
        """
        :param y_true: numpy array of actual glucose values
        :param dy_true: numpy array of actual glucose rate of change
        :param y_pred: numpy array of predicted glucose values
        """
        self.y_true = np.asarray(y_true).flatten()  # Ensure 1D array
        self.dy_true = np.asarray(dy_true).flatten()  # Ensure 1D array
        self.y_pred = np.asarray(y_pred).flatten()  # Ensure 1D array

        # Ensure input consistency
        assert len(self.y_true) == len(self.y_pred) == len(self.dy_true), "Mismatch in input array lengths."

    def full(self):
        """
        Compute P-EGA classifications, matching exactly with metrics.py implementation.
        Returns: numpy array (num_samples, 5), where each row contains a one-hot encoded classification.
        """
        num_samples = len(self.y_true)

        # Calculate dynamic error boundary modification
        mod = np.zeros_like(self.dy_true)
        mod[(self.dy_true >= -2) & (self.dy_true <= -1)] = 10  # Expand upper for falling rates
        mod[(self.dy_true >= 1) & (self.dy_true <= 2)] = 10    # Expand lower for rising rates
        mod[self.dy_true < -2] = 20  # Expand upper for rapid falls
        mod[self.dy_true > 2] = 20   # Expand lower for rapid rises

        # Initialize classification matrix
        classifications = np.zeros((num_samples, 5), dtype=int)  # A, B, C, D, E

        # Define masks for each zone exactly as in metrics.py
        # A zone (Accurate Prediction)
        A_mask = (
            (self.y_pred <= 70 + mod) & (self.y_true <= 70)
        ) | (
            (self.y_pred <= self.y_true * 6/5 + mod) & 
            (self.y_pred >= self.y_true * 4/5 - mod)
        )

        # E zone (Erroneous Reading) - Most critical errors
        E_mask = (
            (self.y_true > 180 + mod) & (self.y_pred < 70 - mod)
        ) | (
            (self.y_pred > 180 + mod) & (self.y_true <= 70 - mod)
        )
        
        # D zone (Dangerous Failure to Detect)
        D_mask = (
            (self.y_pred > 70 + mod) & 
            (self.y_pred > self.y_true * 6/5 + mod) & 
            (self.y_true <= 70) & 
            (self.y_pred <= 180 + mod)
        ) | (
            (self.y_true > 240) & 
            (self.y_pred < 180 - mod) & 
            (self.y_pred >= 70 - mod)
        )
        
        # C zone (Benign Error Leads to Overcorrection)
        C_mask = (
            (self.y_true > 70 + mod) &
            (self.y_pred > self.y_true * 22/17 + (180 - 70 * 22/17) + mod)
        ) | (
            (self.y_true <= 180 - mod) &
            (self.y_pred < self.y_true * 7/10 - 182 - mod)
        )
        
        # Apply masks in the same order as metrics.py to ensure consistency
        classifications[E_mask, 4] = 1  # E
        classifications[D_mask & ~E_mask, 3] = 1  # D
        classifications[C_mask & ~E_mask & ~D_mask, 2] = 1  # C
        classifications[A_mask & ~E_mask & ~D_mask & ~C_mask, 0] = 1  # A
        
        # B zone (Benign Error) - everything else
        B_mask = ~(E_mask | D_mask | C_mask | A_mask)
        classifications[B_mask, 1] = 1  # B
        
        # Ensure one-hot encoding
        assert np.all(np.sum(classifications, axis=1) == 1), "One-hot encoding violated in P-EGA"
        
        return classifications


class R_EGA_Loss:
    """
    The Rate-Error Grid Analysis (R-EGA) estimates the clinical acceptability of glucose predictions 
    based on their rate-of-change accuracy. Implementation matches exactly with metrics.py.
    """
    def __init__(self, dy_true, dy_pred):
        """
        :param dy_true: numpy array of actual glucose rate of change
        :param dy_pred: numpy array of predicted glucose rate of change
        """
        self.dy_true = np.asarray(dy_true).flatten()  # Ensure 1D array
        self.dy_pred = np.asarray(dy_pred).flatten()  # Ensure 1D array

        # Ensure input consistency
        assert len(dy_true) == len(dy_pred), "Mismatch in dy_true and dy_pred lengths."

    def full(self):
        """
        Compute R-EGA classifications ensuring one-hot encoding, matching metrics.py.
        Returns: numpy array with 8 columns (A, B, uC, lC, uD, lD, uE, lE).
        """
        num_samples = len(self.dy_true)

        # Initialize classification matrix
        classifications = np.zeros((num_samples, 8), dtype=int)  # A, B, uC, lC, uD, lD, uE, lE

        # Adaptive threshold based on glucose rate magnitude - exact match to metrics.py
        adaptive_threshold = np.where(np.abs(self.dy_true) >= 4, 2, 1)

        # A zone (Accurate Rate Prediction)
        A_mask = np.abs(self.dy_pred - self.dy_true) <= adaptive_threshold

        # E zones (Erroneous readings) - Applied first for precedence
        uE_mask = (self.dy_pred > 1) & (self.dy_true < -1)  # Upper E: True falling, predicted rising
        lE_mask = (self.dy_pred < -1) & (self.dy_true > 1)  # Lower E: True rising, predicted falling
        
        # D zones (Failure to detect) - Only if not classified as E
        uD_mask = ~uE_mask & (self.dy_pred >= -1) & (self.dy_pred <= 1) & (self.dy_true > self.dy_pred + 2)
        lD_mask = ~lE_mask & (self.dy_pred >= -1) & (self.dy_pred <= 1) & (self.dy_true < self.dy_pred - 2)

        # C zones (Overcorrection) - Benign errors
        uC_mask = (self.dy_true >= -1) & (self.dy_true <= 1) & (self.dy_pred > self.dy_true + 2)
        lC_mask = (self.dy_true >= -1) & (self.dy_true <= 1) & (self.dy_pred < self.dy_true - 2)
        
        # Apply classifications in the same order as metrics.py
        classifications[uE_mask, 6] = 1  # uE
        classifications[lE_mask, 7] = 1  # lE
        classifications[uD_mask, 4] = 1  # uD
        classifications[lD_mask, 5] = 1  # lD
        classifications[uC_mask & ~uE_mask & ~uD_mask, 2] = 1  # uC
        classifications[lC_mask & ~lE_mask & ~lD_mask, 3] = 1  # lC
        classifications[A_mask & ~uE_mask & ~lE_mask & ~uD_mask & ~lD_mask & ~uC_mask & ~lC_mask, 0] = 1  # A
        
        # B zone (Benign Rate Error) - everything else
        B_mask = ~(uE_mask | lE_mask | uD_mask | lD_mask | uC_mask | lC_mask | A_mask)
        classifications[B_mask, 1] = 1  # B
        
        # Ensure one-hot encoding
        assert np.all(np.sum(classifications, axis=1) == 1), "One-hot encoding violated in R-EGA"

        return classifications


class CG_EGA_Loss:
    """
    The Continuous Glucose-Error Grid Analysis (CG-EGA) for loss function use.
    Implementation matches exactly with metrics.py.
    """
    def __init__(self, y_true, dy_true, y_pred, dy_pred, freq=5):
        """
        Args:
            y_true: numpy array of actual glucose values
            dy_true: numpy array of actual glucose rate of change
            y_pred: numpy array of predicted glucose values
            dy_pred: numpy array of predicted glucose rate of change
            freq: prediction frequency in minutes (e.g., 5)
        """
        # Ensure input arrays are properly structured
        self.y_true = np.asarray(y_true).flatten()  # Ensure 1D array
        self.y_pred = np.asarray(y_pred).flatten()  # Ensure 1D array
        self.dy_true = np.asarray(dy_true).flatten()  # Ensure 1D array
        self.dy_pred = np.asarray(dy_pred).flatten()  # Ensure 1D array
        self.freq = freq

        assert len(self.y_true) == len(self.y_pred), f"Length mismatch: y_true ({len(self.y_true)}) != y_pred ({len(self.y_pred)})"
        assert len(self.dy_true) == len(self.dy_pred), f"Length mismatch: dy_true ({len(self.dy_true)}) != dy_pred ({len(self.dy_pred)})"

        # Instantiate P-EGA and R-EGA
        self.p_ega = P_EGA_Loss(self.y_true, self.dy_true, self.y_pred).full()
        self.r_ega = R_EGA_Loss(self.dy_true, self.dy_pred).full()
        
        # Define the P-EGA and R-EGA category labels
        self.p_ega_labels = ["A", "B", "C", "D", "E"]
        self.r_ega_labels = ["A", "B", "uC", "lC", "uD", "lD", "uE", "lE"]
        
        # Define CG-EGA mappings based on provided code
        self.hypo_mapping = {
            # AP: Accurate Predictions
            ("A", "A"): "AP",
            ("A", "B"): "AP",

            # BE: Benign Errors
            ("A", "uC"): "BE",
            ("A", "lC"): "BE",
            ("A", "lD"): "BE",
            ("A", "lE"): "BE",

            # EP: Erroneous Predictions
            ("A", "uD"): "EP",
            ("A", "uE"): "EP",
            ("D", "*"): "EP",
            ("E", "*"): "EP"
        }

        self.eu_mapping = {
            # AP: Accurate Predictions
            ("A", "A"): "AP",
            ("A", "B"): "AP",
            ("B", "A"): "AP",
            ("B", "B"): "AP",

            # BE: Benign Errors
            ("A", "uC"): "BE",
            ("A", "lC"): "BE",
            ("A", "uD"): "BE",
            ("A", "lD"): "BE",
            ("B", "uC"): "BE",
            ("B", "lC"): "BE",
            ("B", "uD"): "BE",
            ("B", "lD"): "BE",

            # EP: Erroneous Predictions
            ("A", "uE"): "EP",
            ("A", "lE"): "EP",
            ("B", "uE"): "EP",
            ("B", "lE"): "EP",
            ("C", "*"): "EP",
        }

        self.hyper_mapping = {
            # AP: Accurate Predictions
            ("A", "A"): "AP",
            ("A", "B"): "AP",
            ("B", "A"): "AP",
            ("B", "B"): "AP",

            # BE: Benign Errors
            ("A", "uC"): "BE",
            ("A", "lC"): "BE",
            ("A", "uD"): "BE",
            ("B", "uC"): "BE",
            ("B", "lC"): "BE",
            ("B", "uD"): "BE",

            # EP: Erroneous Predictions
            ("A", "lD"): "EP",
            ("A", "lE"): "EP",
            ("A", "uE"): "EP",
            ("B", "lD"): "EP",
            ("B", "lE"): "EP",
            ("B", "uE"): "EP",
            ("C", "*"): "EP",
            ("D", "*"): "EP",
            ("E", "*"): "EP"
        }

    def map_cg_ega(self, p_idx, r_idx, glucose_region):
        """
        Maps P-EGA and R-EGA classification to a CG-EGA class (AP, BE, EP)
        based on glucose region, using the exact same logic as in optim_eval_framework.
        
        Args:
            p_idx: Index of P-EGA classification (0-4)
            r_idx: Index of R-EGA classification (0-7)
            glucose_region: String - "hypo", "eu", or "hyper"
        
        Returns:
            String - "AP", "BE", or "EP"
        """
        p_label = self.p_ega_labels[p_idx]
        r_label = self.r_ega_labels[r_idx]
        
        # Select appropriate mapping based on glucose region
        if glucose_region == "hypo":
            mapping = self.hypo_mapping
        elif glucose_region == "eu":
            mapping = self.eu_mapping
        else:  # hyper
            mapping = self.hyper_mapping
            
        # Check for exact match
        if (p_label, r_label) in mapping:
            return mapping[(p_label, r_label)]
        # Check for wildcard P-EGA
        elif (p_label, "*") in mapping:
            return mapping[(p_label, "*")]
        # Check for wildcard R-EGA
        elif ("*", r_label) in mapping:
            return mapping[("*", r_label)]
        # Default to EP if no match is found
        else:
            return "EP"

    def full(self):
        """
        Compute CG-EGA by combining P-EGA and R-EGA, matching metrics.py implementation.

        Returns: CG-EGA matrices for hypoglycemia, euglycemia, and hyperglycemia.
        """
        # Define glycemic regions
        hypoglycemia = (self.y_true <= 70).reshape(-1, 1)
        euglycemia = ((self.y_true > 70) & (self.y_true <= 180)).reshape(-1, 1)
        hyperglycemia = (self.y_true > 180).reshape(-1, 1)

        # Apply region filters (Ensures correct dimensions for np.dot)
        P_hypo = np.concatenate([self.p_ega[:, 0:1], self.p_ega[:, 3:4], self.p_ega[:, 4:5]], axis=1) * hypoglycemia
        P_eu = np.concatenate([self.p_ega[:, 0:1], self.p_ega[:, 1:2], self.p_ega[:, 2:3]], axis=1) * euglycemia
        P_hyper = self.p_ega * hyperglycemia

        R_hypo = self.r_ega * hypoglycemia
        R_eu = self.r_ega * euglycemia
        R_hyper = self.r_ega * hyperglycemia

        # Ensure integer conversion for correct matrix multiplication
        P_hypo, P_eu, P_hyper = P_hypo.astype(int), P_eu.astype(int), P_hyper.astype(int)
        R_hypo, R_eu, R_hyper = R_hypo.astype(int), R_eu.astype(int), R_hyper.astype(int)

        # Compute CG-EGA matrices via dot product, ensuring exact calculation as metrics.py
        CG_EGA_hypo = np.dot(R_hypo.T, P_hypo)  # (8×3)
        CG_EGA_eu = np.dot(R_eu.T, P_eu)  # (8×3)
        CG_EGA_hyper = np.dot(R_hyper.T, P_hyper)  # (8×5)

        assert CG_EGA_hypo.shape == (8, 3), f"CG-EGA Hypo shape incorrect: {CG_EGA_hypo.shape}"
        assert CG_EGA_eu.shape == (8, 3), f"CG-EGA Eu shape incorrect: {CG_EGA_eu.shape}"
        assert CG_EGA_hyper.shape == (8, 5), f"CG-EGA Hyper shape incorrect: {CG_EGA_hyper.shape}"

        return CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper

    def simplified(self):
        """
        Directly maps each sample to AP, BE, or EP classification using the same 
        mapping logic as in optim_eval_framework.
        
        Returns:
            Dictionary of AP, BE, EP counts for hypo, eu, and hyperglycemia regions,
            and a count of samples in each region
        """
        # Get P-EGA and R-EGA class indices
        p_indices = np.argmax(self.p_ega, axis=1)
        r_indices = np.argmax(self.r_ega, axis=1)
        
        # Determine glucose regions for each sample
        hypo_mask = self.y_true <= 70
        eu_mask = (self.y_true > 70) & (self.y_true <= 180)
        hyper_mask = self.y_true > 180
        
        # Classify each sample according to mapping
        classifications = np.array([
            self.map_cg_ega(p, r, "hypo" if hypo_mask[i] else "eu" if eu_mask[i] else "hyper")
            for i, (p, r) in enumerate(zip(p_indices, r_indices))
        ])
        
        # Count occurrences by region and classification
        AP_hypo = np.sum((classifications == "AP") & hypo_mask)
        BE_hypo = np.sum((classifications == "BE") & hypo_mask)
        EP_hypo = np.sum((classifications == "EP") & hypo_mask)
        
        AP_eu = np.sum((classifications == "AP") & eu_mask)
        BE_eu = np.sum((classifications == "BE") & eu_mask)
        EP_eu = np.sum((classifications == "EP") & eu_mask)
        
        AP_hyper = np.sum((classifications == "AP") & hyper_mask)
        BE_hyper = np.sum((classifications == "BE") & hyper_mask)
        EP_hyper = np.sum((classifications == "EP") & hyper_mask)
        
        # Return raw counts (not percentages) to match metrics.py
        results = {
            "AP_hypo": AP_hypo, "BE_hypo": BE_hypo, "EP_hypo": EP_hypo,
            "AP_eu": AP_eu, "BE_eu": BE_eu, "EP_eu": EP_eu,
            "AP_hyper": AP_hyper, "BE_hyper": BE_hyper, "EP_hyper": EP_hyper
        }
        
        counts = {
            "count_hypo": np.sum(hypo_mask), 
            "count_eu": np.sum(eu_mask), 
            "count_hyper": np.sum(hyper_mask)
        }
        
        return results, counts

    def reduced(self):
        """
        Reduces the simplified CG-EGA by not dividing the results into the glycemia regions
        :return: overall AP rate, overall BE rate, overall EP rate
        """
        results, counts = self.simplified()
        sum_counts = counts["count_hypo"] + counts["count_eu"] + counts["count_hyper"]
        
        if sum_counts == 0:
            return 0, 0, 0
            
        sum_AP = results["AP_hypo"] + results["AP_eu"] + results["AP_hyper"]
        sum_BE = results["BE_hypo"] + results["BE_eu"] + results["BE_hyper"]
        sum_EP = results["EP_hypo"] + results["EP_eu"] + results["EP_hyper"]
        
        return sum_AP / sum_counts, sum_BE / sum_counts, sum_EP / sum_counts

    def get_sample_classifications(self):
        """
        Returns the classification (AP, BE, EP) for each individual sample.
        Useful for sample-specific weighting in the loss function.
        
        Returns:
            numpy array of classifications ("AP", "BE", "EP") for each sample
        """
        # Get P-EGA and R-EGA class indices
        p_indices = np.argmax(self.p_ega, axis=1)
        r_indices = np.argmax(self.r_ega, axis=1)
        
        # Determine glucose regions for each sample
        glucose_regions = np.where(
            self.y_true <= 70, "hypo", 
            np.where(self.y_true <= 180, "eu", "hyper")
        )
        
        # Classify each sample according to mapping
        return np.array([
            self.map_cg_ega(p, r, g) 
            for p, r, g in zip(p_indices, r_indices, glucose_regions)
        ])


class CGEGALoss(nn.Module):
    """
    Loss function that uses CG-EGA to weight errors based on clinical significance.
    Updated to use the exact same classification logic as in optim_eval_framework.
    """
    def __init__(self, AP_weight, BE_weight, EP_weight, 
                 hypo_multiplier, 
                 mean=153.064210120389530, std=70.291638454640110, scale_to_glucose=True):
        """
        Args:
            AP_weight: Weight for accurate predictions
            BE_weight: Weight for benign errors
            EP_weight: Weight for erroneous predictions
            hypo_multiplier: Extra penalty multiplier for hypoglycemia errors
            scale_to_glucose: Whether input is normalized and needs scaling
            mean: Mean value for denormalizing glucose values
            std: Standard deviation for denormalizing glucose values
        """
        super(CGEGALoss, self).__init__()
        self.AP_weight = AP_weight
        self.BE_weight = BE_weight
        self.EP_weight = EP_weight
        self.hypo_multiplier = hypo_multiplier
        self.scale_to_glucose = scale_to_glucose
        self.mean = mean
        self.std = std
    
    def forward(self, predictions, targets):
        """
        Compute weighted loss based on CG-EGA classification using the exact same
        mapping logic as in optim_eval_framework.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, 1] or [batch_size, seq_len]
            targets: Target values [batch_size, seq_len, 1] or [batch_size, seq_len]
            
        Returns:
            Weighted RMSE loss
        """
        # Ensure consistent dimensions
        if predictions.dim() > targets.dim():
            predictions = predictions.squeeze(-1)
        if targets.dim() > predictions.dim():
            targets = targets.squeeze(-1)
        
        # Get base squared errors
        squared_errors = (predictions - targets) ** 2
        
        # Convert to glucose values if normalized
        if self.scale_to_glucose:
            pred_glucose = predictions * self.std + self.mean
            target_glucose = targets * self.std + self.mean
        else:
            pred_glucose = predictions
            target_glucose = targets
        
        # Initialize weights tensor
        weights = torch.ones_like(squared_errors)
        
        # Process each batch item individually
        batch_size = predictions.shape[0]
        
        for i in range(batch_size):
            # Convert to CPU numpy for CG-EGA calculation
            y_pred = pred_glucose[i].detach().cpu().numpy().astype(np.float32)
            y_true = target_glucose[i].detach().cpu().numpy().astype(np.float32)
            
            # Calculate derivatives
            dy_pred = np.zeros_like(y_pred, dtype=np.float32)
            dy_true = np.zeros_like(y_true, dtype=np.float32)
            
            if len(y_pred) > 1:
                # Calculate rate of change exactly as in metrics.py
                dy_pred[1:] = (y_pred[1:] - y_pred[:-1]) / np.float32(5.0)
                dy_true[1:] = (y_true[1:] - y_true[:-1]) / np.float32(5.0)
            
            try:
                # Use CG-EGA implementation that matches optim_eval_framework exactly
                cg_ega = CG_EGA_Loss(y_true, dy_true, y_pred, dy_pred, freq=5)
                
                # Get individual sample classifications
                sample_classifications = cg_ega.get_sample_classifications()
                
                # Determine hypo regions for extra weighting
                hypo_mask = y_true <= 70
                
                # Apply weights based on classification
                sample_weights = np.ones_like(y_true, dtype=np.float32)
                
                # Set weights based on classification
                sample_weights[sample_classifications == "AP"] = self.AP_weight
                sample_weights[sample_classifications == "BE"] = self.BE_weight
                sample_weights[sample_classifications == "EP"] = self.EP_weight
                
                # Apply hypoglycemia multiplier
                sample_weights[hypo_mask] *= self.hypo_multiplier
                
                # Transfer weights to tensor
                weights[i] = torch.tensor(sample_weights, device=weights.device, dtype=weights.dtype)
                
            except Exception as e:
                # Fallback to a basic clinical heuristic if the CG-EGA calculation fails
                print(f"CG-EGA calculation failed: {str(e)}. Using fallback heuristic.")
                
                try:
                    # Apply weights based on hypoglycemia, euglycemia, hyperglycemia
                    hypo_mask = y_true <= 70
                    
                    # Calculate absolute and relative errors
                    abs_error = np.abs(y_pred - y_true)
                    rel_error = abs_error / np.maximum(y_true, 1.0)
                    
                    # Classify errors based on clinical significance
                    severe_errors = (rel_error > 0.3) | (abs_error > 40)
                    moderate_errors = (~severe_errors) & ((rel_error > 0.15) | (abs_error > 15))
                    
                    # Apply weights
                    sample_weights = np.ones_like(y_true, dtype=np.float32)
                    sample_weights[severe_errors] = self.EP_weight
                    sample_weights[moderate_errors] = self.BE_weight
                    
                    # Apply hypoglycemia multiplier
                    sample_weights[hypo_mask] *= self.hypo_multiplier
                    
                    # Transfer to tensor
                    weights[i] = torch.tensor(sample_weights, device=weights.device, dtype=weights.dtype)
                    
                except Exception as e:
                    # If even the fallback fails, use a very simple heuristic
                    print(f"Fallback heuristic failed: {str(e)}. Using basic heuristic.")
                    hypo_mask = (target_glucose[i] <= 70).cpu().numpy()
                    sample_weights = np.ones_like(y_true, dtype=np.float32)
                    sample_weights[hypo_mask] = self.EP_weight * self.hypo_multiplier
                    weights[i] = torch.tensor(sample_weights, device=weights.device, dtype=weights.dtype)
        
        # Apply weights and return RMSE
        weighted_squared_errors = squared_errors * weights
        # Handle potential NaN values that might arise from empty sequences
        mean_error = weighted_squared_errors.mean()
        if torch.isnan(mean_error):
            return torch.tensor(0.0, device=mean_error.device, dtype=mean_error.dtype)
        return torch.sqrt(mean_error)





# import torch
# import torch.nn as n
# import torch
# import torch.nn as nn
# import numpy as np






# class P_EGA_Loss:
#     """
#     The Point-Error Grid Analysis (P-EGA) estimates the clinical acceptability of glucose predictions
#     based on their point accuracy. Implementation matches exactly with metrics.py.
#     """
#     def __init__(self, y_true, dy_true, y_pred):
#         """
#         :param y_true: numpy array of actual glucose values
#         :param dy_true: numpy array of actual glucose rate of change
#         :param y_pred: numpy array of predicted glucose values
#         """
#         self.y_true = np.asarray(y_true).flatten()  # Ensure 1D array
#         self.dy_true = np.asarray(dy_true).flatten()  # Ensure 1D array
#         self.y_pred = np.asarray(y_pred).flatten()  # Ensure 1D array

#         # Ensure input consistency
#         assert len(self.y_true) == len(self.y_pred) == len(self.dy_true), "Mismatch in input array lengths."

#     def full(self):
#         """
#         Compute P-EGA classifications, matching exactly with metrics.py implementation.
#         Returns: numpy array (num_samples, 5), where each row contains a one-hot encoded classification.
#         """
#         num_samples = len(self.y_true)

#         # Calculate dynamic error boundary modification
#         mod = np.zeros_like(self.dy_true)
#         mod[(self.dy_true >= -2) & (self.dy_true <= -1)] = 10  # Expand upper for falling rates
#         mod[(self.dy_true >= 1) & (self.dy_true <= 2)] = 10    # Expand lower for rising rates
#         mod[self.dy_true < -2] = 20  # Expand upper for rapid falls
#         mod[self.dy_true > 2] = 20   # Expand lower for rapid rises

#         # Initialize classification matrix
#         classifications = np.zeros((num_samples, 5), dtype=int)  # A, B, C, D, E

#         # Track which samples have been classified to ensure one-hot encoding
#         classified = np.zeros(num_samples, dtype=bool)
        
#         # A zone (Accurate Prediction)
#         A_mask = (
#             (self.y_pred <= 70 + mod) & (self.y_true <= 70)
#         ) | (
#             (self.y_pred <= self.y_true * 1.2 + mod) & 
#             (self.y_pred >= self.y_true * 0.8 - mod)
#         )

#         # E zone (Erroneous Reading) - Most critical errors
#         E_mask = (
#             (self.y_true > 180 + mod) & (self.y_pred < 70 - mod)
#         ) | (
#             (self.y_pred > 180 + mod) & (self.y_true <= 70 - mod)
#         )
        
#         # Apply E zone first (most critical errors)
#         E_effective = E_mask & ~classified
#         classifications[E_effective, 4] = 1
#         classified |= E_effective
        
#         # D zone (Dangerous Failure to Detect)
#         D_mask = (
#             (self.y_pred > 70 + mod) & 
#             (self.y_pred > self.y_true * 1.2 + mod) & 
#             (self.y_true <= 70) & 
#             (self.y_pred <= 180 + mod)
#         ) | (
#             (self.y_true > 240) & 
#             (self.y_pred < 180 - mod) & 
#             (self.y_pred >= 70 - mod)
#         )
        
#         # Apply D zone next
#         D_effective = D_mask & ~classified
#         classifications[D_effective, 3] = 1
#         classified |= D_effective

#         # C zone (Benign Error Leads to Overcorrection)
#         C_mask = (
#             (self.y_true > 70 + mod) &
#             (self.y_pred > self.y_true * 1.3 + mod)
#         ) | (
#             (self.y_true <= 180 - mod) &
#             (self.y_pred < self.y_true * 0.7 - mod)
#         )
        
#         # Apply C zone
#         C_effective = C_mask & ~classified
#         classifications[C_effective, 2] = 1
#         classified |= C_effective
        
#         # A zone (Accurate)
#         A_effective = A_mask & ~classified
#         classifications[A_effective, 0] = 1
#         classified |= A_effective
        
#         # B zone (Benign Error) - everything else
#         B_effective = ~classified
#         classifications[B_effective, 1] = 1
        
#         # Ensure one-hot encoding
#         assert np.all(np.sum(classifications, axis=1) == 1), "One-hot encoding violated in P-EGA"
        
#         return classifications


# class R_EGA_Loss:
#     """
#     The Rate-Error Grid Analysis (R-EGA) estimates the clinical acceptability of glucose predictions 
#     based on their rate-of-change accuracy. Implementation matches exactly with metrics.py.
#     """
#     def __init__(self, dy_true, dy_pred):
#         """
#         :param dy_true: numpy array of actual glucose rate of change
#         :param dy_pred: numpy array of predicted glucose rate of change
#         """
#         self.dy_true = np.asarray(dy_true).flatten()  # Ensure 1D array
#         self.dy_pred = np.asarray(dy_pred).flatten()  # Ensure 1D array

#         # Ensure input consistency
#         assert len(dy_true) == len(dy_pred), "Mismatch in dy_true and dy_pred lengths."

#     def full(self):
#         """
#         Compute R-EGA classifications ensuring one-hot encoding, matching metrics.py.
#         Returns: numpy array with 8 columns (A, B, uC, lC, uD, lD, uE, lE).
#         """
#         num_samples = len(self.dy_true)

#         # Initialize classification matrix
#         classifications = np.zeros((num_samples, 8), dtype=int)

#         # Track which samples have been classified
#         classified = np.zeros(num_samples, dtype=bool)
        
#         # Adaptive threshold based on glucose rate magnitude - exact match to metrics.py
#         adaptive_threshold = np.where(np.abs(self.dy_true) >= 4, 2, 1)

#         # A zone (Accurate Rate Prediction)
#         A_mask = np.abs(self.dy_pred - self.dy_true) <= adaptive_threshold

#         # Erroneous readings (E zones) - Rate estimated in wrong direction
#         uE_mask = (self.dy_pred > 1) & (self.dy_true < -1)  # Upper E: True falling, predicted rising
#         lE_mask = (self.dy_pred < -1) & (self.dy_true > 1)  # Lower E: True rising, predicted falling
        
#         # Apply E zones first (highest priority)
#         uE_effective = uE_mask & ~classified
#         classifications[uE_effective, 6] = 1
#         classified |= uE_effective
        
#         lE_effective = lE_mask & ~classified
#         classifications[lE_effective, 7] = 1
#         classified |= lE_effective

#         # CORRECTION: Fixed the logic for D zones to match metrics.py
#         # Failure to detect (D zones) - Significant change not detected
#         uD_mask = ~uE_mask & (self.dy_pred >= -1) & (self.dy_pred <= 1) & (self.dy_true > self.dy_pred + 2)
#         lD_mask = ~lE_mask & (self.dy_pred >= -1) & (self.dy_pred <= 1) & (self.dy_true < self.dy_pred - 2)
        
#         # Apply D zones
#         uD_effective = uD_mask & ~classified
#         classifications[uD_effective, 4] = 1
#         classified |= uD_effective
        
#         lD_effective = lD_mask & ~classified
#         classifications[lD_effective, 5] = 1
#         classified |= lD_effective

#         # CORRECTION: Fixed the logic for C zones to match metrics.py
#         # Overcorrection (C zones) - No significant true change, but significant predicted change
#         uC_mask = (self.dy_true >= -1) & (self.dy_true <= 1) & (self.dy_pred > self.dy_true + 2)
#         lC_mask = (self.dy_true >= -1) & (self.dy_true <= 1) & (self.dy_pred < self.dy_true - 2)
        
#         # Apply C zones
#         uC_effective = uC_mask & ~classified
#         classifications[uC_effective, 2] = 1
#         classified |= uC_effective
        
#         lC_effective = lC_mask & ~classified
#         classifications[lC_effective, 3] = 1
#         classified |= lC_effective
        
#         # Apply A zone
#         A_effective = A_mask & ~classified
#         classifications[A_effective, 0] = 1
#         classified |= A_effective
        
#         # B zone (Benign Rate Error) - everything else
#         B_effective = ~classified
#         classifications[B_effective, 1] = 1
        
#         # Ensure one-hot encoding
#         assert np.all(np.sum(classifications, axis=1) == 1), "One-hot encoding violated in R-EGA"

#         return classifications


# class CG_EGA_Loss:
#     """
#     The Continuous Glucose-Error Grid Analysis (CG-EGA) for loss function use.
#     Implementation matches exactly with metrics.py.
#     """
#     def __init__(self, y_true, dy_true, y_pred, dy_pred, freq=5):
#         """
#         Args:
#             y_true: numpy array of actual glucose values
#             dy_true: numpy array of actual glucose rate of change
#             y_pred: numpy array of predicted glucose values
#             dy_pred: numpy array of predicted glucose rate of change
#             freq: prediction frequency in minutes (e.g., 5)
#         """
#         # Ensure input arrays are properly structured
#         self.y_true = np.asarray(y_true).flatten()  # Ensure 1D array
#         self.y_pred = np.asarray(y_pred).flatten()  # Ensure 1D array
#         self.dy_true = np.asarray(dy_true).flatten()  # Ensure 1D array
#         self.dy_pred = np.asarray(dy_pred).flatten()  # Ensure 1D array
#         self.freq = freq

#         assert len(self.y_true) == len(self.y_pred), f"Length mismatch: y_true ({len(self.y_true)}) != y_pred ({len(self.y_pred)})"
#         assert len(self.dy_true) == len(self.dy_pred), f"Length mismatch: dy_true ({len(self.dy_true)}) != dy_pred ({len(self.dy_pred)})"

#         # Instantiate P-EGA and R-EGA
#         self.p_ega = P_EGA_Loss(self.y_true, self.dy_true, self.y_pred).full()
#         self.r_ega = R_EGA_Loss(self.dy_true, self.dy_pred).full()

#     def full(self):
#         """
#         Compute CG-EGA by combining P-EGA and R-EGA, matching metrics.py implementation.

#         Returns: CG-EGA matrices for hypoglycemia, euglycemia, and hyperglycemia.
#         """
#         # Define glycemic regions
#         hypoglycemia = (self.y_true <= 70).reshape(-1, 1)
#         euglycemia = ((self.y_true > 70) & (self.y_true <= 180)).reshape(-1, 1)
#         hyperglycemia = (self.y_true > 180).reshape(-1, 1)

#         # Apply region filters (Ensures correct dimensions for np.dot)
#         P_hypo = np.concatenate([self.p_ega[:, 0:1], self.p_ega[:, 3:4], self.p_ega[:, 4:5]], axis=1) * hypoglycemia
#         P_eu = np.concatenate([self.p_ega[:, 0:1], self.p_ega[:, 1:2], self.p_ega[:, 2:3]], axis=1) * euglycemia
#         P_hyper = self.p_ega * hyperglycemia

#         R_hypo = self.r_ega * hypoglycemia
#         R_eu = self.r_ega * euglycemia
#         R_hyper = self.r_ega * hyperglycemia

#         # Ensure integer conversion for correct matrix multiplication
#         P_hypo, P_eu, P_hyper = P_hypo.astype(int), P_eu.astype(int), P_hyper.astype(int)
#         R_hypo, R_eu, R_hyper = R_hypo.astype(int), R_eu.astype(int), R_hyper.astype(int)

#         # Compute CG-EGA matrices via dot product, ensuring exact calculation as metrics.py
#         CG_EGA_hypo = np.dot(R_hypo.T, P_hypo)  # (8×3)
#         CG_EGA_eu = np.dot(R_eu.T, P_eu)  # (8×3)
#         CG_EGA_hyper = np.dot(R_hyper.T, P_hyper)  # (8×5)

#         assert CG_EGA_hypo.shape == (8, 3), f"CG-EGA Hypo shape incorrect: {CG_EGA_hypo.shape}"
#         assert CG_EGA_eu.shape == (8, 3), f"CG-EGA Eu shape incorrect: {CG_EGA_eu.shape}"
#         assert CG_EGA_hyper.shape == (8, 5), f"CG-EGA Hyper shape incorrect: {CG_EGA_hyper.shape}"

#         return CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper

#     def simplified(self):
#         """
#         Simplifies CG-EGA into Accurate Prediction (AP), Benign Prediction (BE), 
#         and Erroneous Prediction (EP) rates for every glycemic region.
#         Implementation matches metrics.py.
        
#         Returns:
#             Dictionary of AP, BE, EP rates for hypo, eu, and hyperglycemia, 
#             and a count of samples in each region
#         """
#         # This method follows exactly the same implementation as in metrics.py
#         CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper = self.full()

#         # Define filter definitions exactly as in metrics.py
#         # For hypoglycemia
#         filter_AP_hypo = np.array([
#             [1, 0, 0],
#             [1, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#         ])

#         filter_BE_hypo = np.array([
#             [0, 0, 0],
#             [0, 0, 0],
#             [1, 0, 0],
#             [1, 0, 0],
#             [0, 0, 0],
#             [1, 0, 0],
#             [0, 0, 0],
#             [1, 0, 0],
#         ])

#         filter_EP_hypo = np.array([
#             [0, 1, 1],
#             [0, 1, 1],
#             [0, 1, 1],
#             [0, 1, 1],
#             [1, 1, 1],
#             [0, 1, 1],
#             [1, 1, 1],
#             [0, 1, 1],
#         ])

#         # For euglycemia
#         filter_AP_eu = np.array([
#             [1, 1, 0],
#             [1, 1, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#         ])

#         filter_BE_eu = np.array([
#             [0, 0, 0],
#             [0, 0, 0],
#             [1, 1, 0],
#             [1, 1, 0],
#             [1, 1, 0],
#             [1, 1, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#         ])

#         filter_EP_eu = np.array([
#             [0, 0, 1],
#             [0, 0, 1],
#             [0, 0, 1],
#             [0, 0, 1],
#             [0, 0, 1],
#             [0, 0, 1],
#             [1, 1, 1],
#             [1, 1, 1],
#         ])

#         # For hyperglycemia
#         filter_AP_hyper = np.array([
#             [1, 1, 0, 0, 0],
#             [1, 1, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#         ])

#         filter_BE_hyper = np.array([
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#             [1, 1, 0, 0, 0],
#             [1, 1, 0, 0, 0],
#             [1, 1, 0, 0, 0],
#             [1, 1, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#         ])

#         filter_EP_hyper = np.array([
#             [0, 0, 1, 1, 1],
#             [0, 0, 1, 1, 1],
#             [0, 0, 1, 1, 1],
#             [0, 0, 1, 1, 1],
#             [0, 0, 1, 1, 1],
#             [0, 0, 1, 1, 1],
#             [1, 1, 1, 1, 1],
#             [1, 1, 1, 1, 1],
#         ])

#         # Count samples in each glycemic region
#         hypoglycemia = (self.y_true <= 70).reshape(-1, 1)
#         euglycemia = ((self.y_true > 70) & (self.y_true <= 180)).reshape(-1, 1)
#         hyperglycemia = (self.y_true > 180).reshape(-1, 1)
        
#         hypo_count = np.sum(hypoglycemia)
#         eu_count = np.sum(euglycemia)
#         hyper_count = np.sum(hyperglycemia)

#         # Calculate AP, BE, EP values using matrix operations, matching metrics.py
#         AP_hypo = np.sum(CG_EGA_hypo * filter_AP_hypo)
#         BE_hypo = np.sum(CG_EGA_hypo * filter_BE_hypo)
#         EP_hypo = np.sum(CG_EGA_hypo * filter_EP_hypo)

#         AP_eu = np.sum(CG_EGA_eu * filter_AP_eu)
#         BE_eu = np.sum(CG_EGA_eu * filter_BE_eu)
#         EP_eu = np.sum(CG_EGA_eu * filter_EP_eu)

#         AP_hyper = np.sum(CG_EGA_hyper * filter_AP_hyper)
#         BE_hyper = np.sum(CG_EGA_hyper * filter_BE_hyper)
#         EP_hyper = np.sum(CG_EGA_hyper * filter_EP_hyper)

#         # Return all results as a dictionary, matching metrics.py
#         results = {
#             "AP_hypo": AP_hypo, "BE_hypo": BE_hypo, "EP_hypo": EP_hypo,
#             "AP_eu": AP_eu, "BE_eu": BE_eu, "EP_eu": EP_eu,
#             "AP_hyper": AP_hyper, "BE_hyper": BE_hyper, "EP_hyper": EP_hyper
#         }
        
#         counts = {"count_hypo": hypo_count, "count_eu": eu_count, "count_hyper": hyper_count}
        
#         # These are raw counts, not percentages
#         return results, counts

#     def reduced(self):
#         """
#         Reduces the simplified CG-EGA by not dividing the results into the glycemia regions
#         :return: overall AP rate, overall BE rate, overall EP rate
#         """
#         results, counts = self.simplified()
#         sum_counts = counts["count_hypo"] + counts["count_eu"] + counts["count_hyper"]
        
#         if sum_counts == 0:
#             return 0, 0, 0
            
#         sum_AP = results["AP_hypo"] + results["AP_eu"] + results["AP_hyper"]
#         sum_BE = results["BE_hypo"] + results["BE_eu"] + results["BE_hyper"]
#         sum_EP = results["EP_hypo"] + results["EP_eu"] + results["EP_hyper"]
        
#         return sum_AP / sum_counts, sum_BE / sum_counts, sum_EP / sum_counts


# class CGEGALoss(nn.Module):
#     """
#     Loss function that uses CG-EGA to weight errors based on clinical significance.
#     Implementation follows Kovatchev et al. (2004) and matches metrics.py.
#     """
#     def __init__(self, AP_weight=1.0, BE_weight=2.0, EP_weight=6.0, 
#                  hypo_multiplier=3.0, scale_to_glucose=True, 
#                  mean=153.064210120389530, std=70.291638454640110):
#         """
#         Args:
#             AP_weight: Weight for accurate predictions
#             BE_weight: Weight for benign errors
#             EP_weight: Weight for erroneous predictions
#             hypo_multiplier: Extra penalty multiplier for hypoglycemia errors
#             scale_to_glucose: Whether input is normalized and needs scaling
#             mean: Mean value for denormalizing glucose values
#             std: Standard deviation for denormalizing glucose values
#         """
#         super(CGEGALoss, self).__init__()
#         self.AP_weight = AP_weight
#         self.BE_weight = BE_weight
#         self.EP_weight = EP_weight
#         self.hypo_multiplier = hypo_multiplier
#         self.scale_to_glucose = scale_to_glucose
#         self.mean = mean
#         self.std = std
    
#     def forward(self, predictions, targets):
#         """
#         Compute weighted loss based on CG-EGA classification.
        
#         Args:
#             predictions: Model predictions [batch_size, seq_len, 1] or [batch_size, seq_len]
#             targets: Target values [batch_size, seq_len, 1] or [batch_size, seq_len]
            
#         Returns:
#             Weighted RMSE loss
#         """
#         # Ensure consistent dimensions
#         if predictions.dim() > targets.dim():
#             predictions = predictions.squeeze(-1)
#         if targets.dim() > predictions.dim():
#             targets = targets.squeeze(-1)
        
#         # Get base squared errors
#         squared_errors = (predictions - targets) ** 2
        
#         # Convert to glucose values if normalized
#         if self.scale_to_glucose:
#             pred_glucose = predictions * self.std + self.mean
#             target_glucose = targets * self.std + self.mean
#         else:
#             pred_glucose = predictions
#             target_glucose = targets
        
#         # Initialize weights tensor
#         weights = torch.ones_like(squared_errors)
        
#         # Process each batch item individually
#         batch_size = predictions.shape[0]
        
#         for i in range(batch_size):
#             # Convert to CPU numpy for CG-EGA calculation
#             y_pred = pred_glucose[i].detach().cpu().numpy().astype(np.float32)
#             y_true = target_glucose[i].detach().cpu().numpy().astype(np.float32)
            
#             # Calculate derivatives - ensure consistency with metrics.py
#             dy_pred = np.zeros_like(y_pred, dtype=np.float32)
#             dy_true = np.zeros_like(y_true, dtype=np.float32)
            
#             if len(y_pred) > 1:
#                 # Ensure rate calculations match exactly how metrics.py does it
#                 dy_pred[1:] = (y_pred[1:] - y_pred[:-1]) / np.float32(5.0)
#                 dy_true[1:] = (y_true[1:] - y_true[:-1]) / np.float32(5.0)
            
#             try:
#                 # Use CG-EGA implementation that matches metrics.py exactly
#                 cg_ega = CG_EGA_Loss(y_true, dy_true, y_pred, dy_pred, freq=5)
#                 results, counts = cg_ega.simplified()
                
#                 # Create sample-specific error maps rather than region-wide weights
#                 hypo_mask = y_true <= 70
#                 eu_mask = (y_true > 70) & (y_true <= 180)
#                 hyper_mask = y_true > 180
                
#                 # Improved method to map CG-EGA results to individual samples
#                 # Create individual P-EGA and R-EGA classifications for each sample
#                 p_ega = P_EGA_Loss(y_true, dy_true, y_pred).full()
#                 r_ega = R_EGA_Loss(dy_true, dy_pred).full()
                
#                 # Initialize sample-specific weights
#                 sample_weights = np.ones_like(y_true, dtype=np.float32)
                
#                 # For each sample, determine its error category based on P-EGA and R-EGA
#                 for j in range(len(y_true)):
#                     # Skip NaN values
#                     if np.isnan(y_true[j]) or np.isnan(y_pred[j]):
#                         continue
                    
#                     # Get P-EGA and R-EGA classes for this sample
#                     p_class = np.argmax(p_ega[j])
#                     r_class = np.argmax(r_ega[j])
                    
#                     # Apply weights based on glycemic region and error category
#                     if hypo_mask[j]:
#                         # For hypoglycemia
#                         if p_class == 0 and (r_class == 0 or r_class == 1):  # AP
#                             sample_weights[j] = self.AP_weight * self.hypo_multiplier
#                         elif ((p_class == 0) and (r_class in [2, 3, 5, 7])) or \
#                              ((p_class == 3) and (r_class == 1)):  # BE
#                             sample_weights[j] = self.BE_weight * self.hypo_multiplier
#                         else:  # EP
#                             sample_weights[j] = self.EP_weight * self.hypo_multiplier
                    
#                     elif eu_mask[j]:
#                         # For euglycemia
#                         if ((p_class == 0) and (r_class == 0 or r_class == 1)) or \
#                            ((p_class == 1) and (r_class == 0 or r_class == 1)):  # AP
#                             sample_weights[j] = self.AP_weight
#                         elif ((p_class in [0, 1]) and (r_class in [2, 3, 4, 5])) or \
#                              ((p_class == 2) and (r_class in [0, 1, 2, 3])):  # BE
#                             sample_weights[j] = self.BE_weight
#                         else:  # EP
#                             sample_weights[j] = self.EP_weight
                    
#                     elif hyper_mask[j]:
#                         # For hyperglycemia
#                         if ((p_class == 0 or p_class == 1) and (r_class == 0 or r_class == 1)):  # AP
#                             sample_weights[j] = self.AP_weight
#                         elif ((p_class in [0, 1]) and (r_class in [2, 3, 4, 5])) or \
#                              ((p_class in [2, 3]) and (r_class in [0, 1, 2, 3])):  # BE
#                             sample_weights[j] = self.BE_weight
#                         else:  # EP
#                             sample_weights[j] = self.EP_weight
                
#                 # Safe tensor device handling
#                 weights[i] = torch.tensor(sample_weights, device=weights.device, dtype=weights.dtype)
                
#             except Exception as e:
#                 # Fallback to improved heuristic if CG-EGA calculation fails
#                 print(f"CG-EGA calculation failed: {str(e)}. Using fallback heuristic.")
                
#                 # More robust fallback using a clinically-oriented heuristic
#                 try:
#                     # Apply weights based on hypoglycemia, euglycemia, hyperglycemia
#                     hypo_mask = torch.tensor(y_true <= 70, device=weights.device)
#                     eu_mask = torch.tensor((y_true > 70) & (y_true <= 180), device=weights.device)
#                     hyper_mask = torch.tensor(y_true > 180, device=weights.device)
                    
#                     # Calculate absolute and relative errors
#                     abs_error = torch.abs(pred_glucose[i] - target_glucose[i])
#                     rel_error = abs_error / torch.clamp(target_glucose[i], min=1.0)
                    
#                     # Classify errors based on clinical significance
#                     severe_errors = (rel_error > 0.3) | (abs_error > 40)
#                     moderate_errors = (~severe_errors) & ((rel_error > 0.15) | (abs_error > 15))
                    
#                     # Apply weights
#                     weights[i] = torch.ones_like(weights[i])
#                     weights[i][severe_errors] = self.EP_weight
#                     weights[i][moderate_errors] = self.BE_weight
                    
#                     # Apply hypoglycemia multiplier
#                     weights[i][hypo_mask] *= self.hypo_multiplier
                    
#                 except Exception as e:
#                     # If even the fallback fails, use a very simple clinical heuristic
#                     print(f"Fallback heuristic failed: {str(e)}. Using basic heuristic.")
#                     hypo_mask = (target_glucose[i] <= 70)
#                     weights[i][hypo_mask] = self.EP_weight * self.hypo_multiplier
        
#         # Apply weights and return RMSE
#         weighted_squared_errors = squared_errors * weights
#         # Handle potential NaN values that might arise from empty sequences
#         mean_error = weighted_squared_errors.mean()
#         if torch.isnan(mean_error):
#             return torch.tensor(0.0, device=mean_error.device, dtype=mean_error.dtype)
#         return torch.sqrt(mean_error)