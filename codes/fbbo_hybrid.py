import numpy as np
import random
import time
import matplotlib.pyplot as plt

# === Αντικείμενο Solution ===
class solution:
    def __init__(self):
        self.best = None
        self.convergence = []
        self.optimizer = ""
        self.objfname = ""
        self.executionTime = 0
        self.startTime = ""
        self.endTime = ""
        self.visited_positions = []  

# === ClearDups dummy ===
def ClearDups(pos, pop_size, dim, ub, lb):
    return np.clip(pos + 1e-8 * np.random.randn(*pos.shape), lb, ub)

# ===Συνάρτηση Κόστους ===
def objective_func(x):
    return 5

# === FWA βήμα ===
def fa_step(fitness_function, fireworks, fitnesses, lwr_bnd, upp_bnd, m=50, big_a_hat=40, a=0.04, b=0.8, mg=5):
    n, d = fireworks.shape
    epsilon = np.finfo(float).eps
    all_sparks = np.array(fireworks)

    for i in range(n):
        si = m * (np.max(fitnesses) - fitnesses[i] + epsilon) / (np.sum(np.max(fitnesses) - fitnesses) + epsilon)
        si = int(round(np.clip(si, a * m, b * m)))

        ai = big_a_hat * (fitnesses[i] - np.min(fitnesses) + epsilon) / (np.sum(fitnesses - np.min(fitnesses)) + epsilon)
        sparks_i = np.zeros((si, d))

        for s in range(si):
            sparks_i[s, :] = fireworks[i, :]
            z = np.random.choice(d, size=np.random.randint(1, d+1), replace=False)
            h = ai * np.random.uniform(-1, 1)
            sparks_i[s, z] += h
            sparks_i[s] = np.clip(sparks_i[s], lwr_bnd, upp_bnd)

        all_sparks = np.vstack((all_sparks, sparks_i))

    # Gaussian Sparks
    idx = np.random.choice(range(len(all_sparks)), mg, replace=True)
    gaussian_sparks = np.copy(all_sparks[idx, :])
    for i in range(mg):
        z = np.random.choice(d, size=np.random.randint(1, d+1), replace=False)
        g = np.random.normal(1, 1)
        gaussian_sparks[i, z] *= g
        gaussian_sparks[i] = np.clip(gaussian_sparks[i], lwr_bnd, upp_bnd)

    all_sparks = np.vstack((gaussian_sparks, all_sparks))
    return all_sparks

# === Hybrid FBBO-FWA ===
def hybrid_bbo_fwa(objf, lb, ub, dim, PopSize=50, iters=100, Keep=2):
    s = solution()
    pos = np.random.uniform(lb, ub, (PopSize, dim))
    fit = np.apply_along_axis(objf, 1, pos)
    mu = np.array([(PopSize + 1 - i) / (PopSize + 1) for i in range(PopSize)])
    lam = 1 - mu
    pmutate = 0.01

    MinCost = np.zeros(iters)
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    all_positions = []

    for l in range(iters):
        # BBO Migration
        Island = np.copy(pos)
        for k in range(PopSize):
            for j in range(dim):
                if random.random() < lam[k]:
                    RandomNum = random.random() * sum(mu)
                    Select = mu[0]
                    SelectIndex = 0
                    while RandomNum > Select and SelectIndex < PopSize - 1:
                        SelectIndex += 1
                        Select += mu[SelectIndex]
                    Island[k, j] = pos[SelectIndex, j]

        # Mutation
        for k in range(PopSize):
            for j in range(dim):
                if random.random() < pmutate:
                    Island[k, j] = lb[j] + random.random() * (ub[j] - lb[j])

        Island = np.clip(Island, lb, ub)

        # FWA step
        Island = fa_step(objf, Island, np.apply_along_axis(objf, 1, Island), lb, ub)

        # Καταγραφή θέσεων
        all_positions.extend(Island.tolist())

        # Elitism
        fitnesses = np.apply_along_axis(objf, 1, Island)
        idx = np.argsort(fitnesses)
        Island = Island[idx, :]
        fitnesses = fitnesses[idx]
        pos = Island[:PopSize, :]
        fit = fitnesses[:PopSize]

        pos = ClearDups(pos, PopSize, dim, ub, lb)
        fit = np.apply_along_axis(objf, 1, pos)
        MinCost[l] = np.min(fit)
        
        if l % 10 == 0 or l == iters - 1:
            print(f"Iteration {l+1}: Best = {MinCost[l]}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = MinCost
    s.optimizer = "Hybrid BBO + FWA"
    s.objfname = objf.__name__
    s.best = np.min(fit)
    s.visited_positions = np.array(all_positions)
    return s

# --- Παραμετρικός ορισμός προβλήματος
dim = 2
lb = np.array([-5] * dim)
ub = np.array([5] * dim)

# --- Εκτέλεση υβριδικού αλγορίθμου
s = hybrid_bbo_fwa(objective_func, lb, ub, dim, PopSize=20, iters=100)

# --- Εκτύπωση τελικού αποτελέσματος
print("\nFinal best fitness:", s.best)
print("Execution time (sec):", round(s.executionTime, 3))

# --- Γράφημα σύγκλισης
try:
    plt.plot(s.convergence, label="Hybrid BBO + FWA")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Convergence Curve")
    plt.grid(True)
    plt.legend()
    plt.show()
except:
    pass

# === Signature Plot ===
def plot_signature(points, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5, c=np.random.rand(len(points)))
    plt.title(title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.tight_layout()
    plt.show()

# === Signature Data Generator ===
def generate_signature_points(runs=100, samples_per_run=1000):
    all_points = []
    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        result = hybrid_bbo_fwa(objective_func, lb=np.array([-5]*2), ub=np.array([5]*2), dim=2, PopSize=10, iters=5)
        visited = result.visited_positions
        if len(visited) >= samples_per_run:
            points = visited[np.random.choice(visited.shape[0], samples_per_run, replace=False)]
        else:
            points = visited
        all_points.append(points)
    return np.vstack(all_points)

# === Κύρια Εκτέλεση ===
if __name__ == "__main__":
    sig_points = generate_signature_points(runs=100, samples_per_run=1000)
    plot_signature(sig_points, "Signature Plot FBBO")
