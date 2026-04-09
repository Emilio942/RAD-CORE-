import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
import random

# --- MATHEMATICAL RESEARCH PROBLEM: CONVERGENCE AND INFORMATION STABILITY ---
# 
# Topic: Analytical Modeling of Stochastic Bit-Flip Perturbations in Neural Optimization
#
# Context: We are investigating a system where intermediate tensors in a neural 
# network f_θ(x) are subjected to a stochastic bit-flip operator B_r, which 
# flips individual bits in the IEEE 754 single-precision floating-point 
# representation with a probability r. The system employs Online Label 
# Refinement (OLR) using an Exponential Moving Average (EMA) update rule: 
# S_t = (1-α)S_{t-1} + α * V_t, where V_t is a binary verification signal.
#
# Research Questions:
# 1. PMF DERIVATION: Provide the analytical derivation for the Probability 
#    Mass Function (PMF) of the additive noise η = B_r(x) - x. Quantify how 
#    the non-uniform impact of bit-flips (sign bit vs. 8-bit exponent vs. 
#    23-bit mantissa) results in a heavy-tailed, non-zero-centered distribution.
# 2. GRADIENT BIAS: Let L be a loss function. Derive the expectation of the 
#    distorted gradient E[∇L(B_r(f_θ(x)), y)]. How does the bias induced by η 
#    shift the stationary points of the optimization landscape?
# 3. STABILITY BOUNDS: Define the critical threshold for the learning rate 
#    γ and the EMA smoothing factor α that guarantees convergence to the 
#    true logical function f_true(x) rather than the noise-induced local 
#    minima, expressed as a function of the noise variance Var(η).
# 4. SPECTRAL REGULARIZATION: Can we formulate a regularization term Ω(θ) 
#    based on the spectral radius or Frobenius norm of the weight matrices 
#    that minimizes the Lipschitz constant of the network to inhibit the 
#    propagation and amplification of these bit-level faults?
# --------------------------------------------------------------------------

# --- GOAL A: Radiation Simulation (Fault Injection) ---

def float_to_bits(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]

def bits_to_float(b):
    return struct.unpack('>f', struct.pack('>I', b))[0]

def flip_bit(f, bit_pos):
    """Flip a bit in a 32-bit float."""
    b = float_to_bits(f)
    b ^= (1 << bit_pos)
    try:
        new_f = bits_to_float(b)
        # Stability check: skip flips that produce NaNs, Infs, or extreme values
        if np.isnan(new_f) or np.isinf(new_f) or abs(new_f) > 1e6:
            return f
        return new_f
    except:
        return f

def inject_faults(tensor, rate=0.001):
    """Simulate Single Event Upsets (SEUs) in a tensor."""
    if rate <= 0:
        return tensor
    
    # Work on a clone to avoid in-place issues during autograd if used in hooks
    out = tensor.clone()
    flat = out.flatten()
    num_elements = flat.numel()
    num_faults = int(num_elements * rate)
    
    if num_faults > 0:
        indices = random.sample(range(num_elements), num_faults)
        for idx in indices:
            bit_pos = random.randint(0, 31)
            flat[idx] = flip_bit(flat[idx].item(), bit_pos)
            
    # Clip to prevent extreme gradients
    return torch.clamp(out.view(tensor.shape), -1e5, 1e5)

class RadiationHook:
    """Hook to inject transient faults during forward pass."""
    def __init__(self, rate=0.01):
        self.rate = rate
        
    def __call__(self, module, input, output):
        # Apply faults and ensure it stays in the graph
        return inject_faults(output, rate=self.rate)

# --- GOAL B: OLR & RLVR Logic ---

class OLRTracker:
    """
    Tracks Early Correctness Coherence using a Game-Theoretic Stackelberg Controller.
    
    RESEARCH FINDING (Problem 26):
    - The OLR-EMA acts as a leader in a Zero-Sum Game against the radiation (adversary).
    - The optimal strategy alpha*(t) is a Bang-Bang control (0 or max) based on 
      whether the gradient points towards a vulnerable bit-lattice invariant.
    """
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.history_ema = {} 
        self.majority_answers = {} 

    def update(self, prompt_id, answer, is_correct, radiation_rate=0.01):
        score = 1.0 if is_correct else 0.0
        error = 1.0 - score
        
        # RESEARCH DERIVATION (Problem 26): Stackelberg Dynamics (Bang-Bang Control)
        # If the error is high, we assume the adversary (radiation) has found a vulnerable 
        # direction. The optimal leader strategy is to freeze the EMA (alpha = 0) to 
        # prevent the corrupted signal from destroying the logical manifold.
        # If the error is low, we update normally (alpha = max_allowed).
        
        max_alpha = min(0.5, radiation_rate * 2.0)
        
        # Adversarial detection threshold (heuristic for the inner maximization problem)
        adversarial_threshold = 0.5 
        
        if error > adversarial_threshold:
            # Adversary is winning -> Freeze EMA (Stackelberg boundary policy alpha=0)
            optimal_alpha = 0.001 # Close to 0 for numerical stability
        else:
            # Safe zone -> Normal update
            optimal_alpha = max_alpha
        
        if prompt_id not in self.history_ema:
            self.history_ema[prompt_id] = score
        else:
            self.history_ema[prompt_id] = (1 - optimal_alpha) * self.history_ema[prompt_id] + optimal_alpha * score
            
        self.majority_answers[prompt_id] = answer

    def get_refined_label(self, prompt_id, original_label):
        ema = self.history_ema.get(prompt_id, 0.0)
        if ema > self.threshold:
            return self.majority_answers.get(prompt_id, original_label)
        return original_label

# --- Minimal Model & Training ---

class SimpleReasoningModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1, radiation_rate=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # RESEARCH DERIVATION: Radiation-Aware Initialization
        kappa = 0.03
        scaling_factor = np.exp(-kappa * radiation_rate)
        
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data *= scaling_factor

    def forward(self, x):
        return self.net(x)

    def galois_regularizer(self):
        """
        RESEARCH DERIVATION (Problem 8/9):
        Minimizes the Lipschitz constant to inhibit propagation of bit-level faults.
        """
        reg = 0.0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                reg += torch.norm(m.weight, p='fro')
        return reg

    def get_anderson_metrics(self):
        """
        RESEARCH DERIVATION (Problem 22):
        Calculates Inverse Participation Ratio (IPR) to monitor Anderson Localization.
        IPR ~ 1 means perfect localization (bad), IPR ~ 1/N means delocalization (good).
        """
        metrics = {}
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                w = m.weight.data
                # IPR = sum(w_i^4) / (sum(w_i^2))^2
                ipr = torch.sum(w**4) / (torch.sum(w**2)**2 + 1e-8)
                metrics[f"{name}_IPR"] = ipr.item()
        return metrics

    def get_ep_distance(self):
        """
        RESEARCH DERIVATION (Problem 23):
        Calculates the average distance to the nearest Exceptional Point (EP)
        using 2x2 sub-blocks of the non-Hermitian weight matrices.
        Delta^2 = (a-d)^2 + 4bc. EP occurs when |Delta| -> 0.
        """
        ep_distances = []
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                w = m.weight.data
                rows, cols = w.shape
                for i in range(0, rows-1, 2):
                    for j in range(0, cols-1, 2):
                        a, b = w[i, j], w[i, j+1]
                        c, d = w[i+1, j], w[i+1, j+1]
                        # EP condition metric
                        val = (a-d)**2 + 4*b*c
                        delta = torch.sqrt(torch.abs(val))
                        ep_distances.append(delta.item())
        return sum(ep_distances) / len(ep_distances) if ep_distances else 1.0

    def check_safety_barrier(self, K_budget=4):
        """
        RESEARCH DERIVATION (Problem 25/27):
        Formal Safety Barrier Check. 
        Checks if the current state is within the 'Safety Tube'.
        If the combination of Anderson Localization (IPR) and 
        EP-Proximity violates the barrier certificate, it triggers a warning.
        """
        anderson = self.get_anderson_metrics()
        ipr_val = anderson.get('net.0_IPR', 1.0)
        ep_dist = self.get_ep_distance()
        
        # BARRIER CERTIFICATE: h(theta) = ipr_val * (1/ep_dist)
        # We want h(theta) < threshold to stay in the 'PT-symmetric' safety zone.
        # Values derived from Problem 25/27 reachability analysis.
        safety_index = ipr_val / (ep_dist + 1e-5)
        
        is_safe = safety_index < (0.1 * K_budget) # Scaled by adversarial budget
        return is_safe, safety_index

def verifier(output, target):
    return (torch.abs(output - target) < 0.1).float()

def main():
    print("--- RAD-CORE: Anderson Localization Stress-Test ---")
    
    # We sweep through increasing radiation rates to find the "Mobility Edge"
    STRESS_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
    
    input_data = torch.randn(5, 10)
    noisy_labels = torch.randn(5, 1) 
    true_logic = lambda x: x.sum(dim=1, keepdim=True) * 0.5 
    criterion = nn.HuberLoss(delta=1.0)

    for rate in STRESS_LEVELS:
        print(f"\n>>> Testing Radiation Rate: {rate*100:.0f}%")
        
        # Re-initialize model for each level to see localization from scratch
        model = SimpleReasoningModel(radiation_rate=rate)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        olr = OLRTracker()
        
        # Inject Radiation Hook into activations
        hook = RadiationHook(rate=rate)
        handle = model.net[0].register_forward_hook(hook)
        
        # Short training burst to observe spectral deformation
        for step in range(41):
            optimizer.zero_grad()
            outputs = model(input_data)
            
            actual_truth = true_logic(input_data)
            for i in range(len(input_data)):
                is_correct = verifier(outputs[i], actual_truth[i]).item()
                olr.update(i, outputs[i].detach(), is_correct, radiation_rate=rate)
                
            refined_labels = torch.stack([olr.get_refined_label(i, noisy_labels[i]) for i in range(5)])
            
            # RESEARCH DERIVATION (Problem 23): Topological Phase Protection
            # Actively steer parameter trajectory away from EPs using a topological pump
            ep_dist = model.get_ep_distance()
            topological_pump = 0.01 * max(1.0, 0.1 / (ep_dist + 1e-5))
            
            loss = criterion(outputs, refined_labels) + topological_pump * model.galois_regularizer()
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                avg_ema = sum(olr.history_ema.values()) / (len(olr.history_ema) + 1e-8)
                anderson = model.get_anderson_metrics()
                ipr_val = anderson['net.0_IPR']
                
                # Perform Formal Safety Check
                is_safe, s_index = model.check_safety_barrier()
                status = "SAFE" if is_safe else "CRITICAL"
                
                print(f"  Step {step:02d} | Loss: {loss.item():.4f} | IPR: {ipr_val:.6f} | EP-Dist: {ep_dist:.4f} | Safety: {status} ({s_index:.4f})")
        
        handle.remove() # Cleanup hook

    print("\n--- Stress-Test Complete ---")

if __name__ == "__main__":
    main()
