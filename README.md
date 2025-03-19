# ML
Task 1. Implementation of the calculation of the Minkowski distance for a given set of points relative to a reference point.

Minkowski distance calculation function:
def minkowski_distance(x, y, p):
    return np.sum(np.abs(x - y) ** p, axis=1) ** (1 / p)

    Minkowski distance is a generalized metric that can take the form of Manhattan or Euclidean distance depending on p
    p = 1 → Manhattan distance
    p = 2 → Euclidean distance
    p → ∞ → Chebyshev distance

 To calculate, we also need to generate a reference point and points for analyzing the behavior of the distance
 //
        points = np.array([
         [0, 0],
         [0, 1],
         [1, 0],
         [1, 1]
])
     reference_point = np.array([0.5, 0.5])
//

points – an array of coordinates of 4 points located in a square.
reference_point – the central point (0.5, 0.5) relative to which distances are measured.

Visualization of distances in the form of circles of different radii.
The radius depends on p:

p=1 → square shape (Manhattan distance)

p=2 → round shape (Euclidean distance)

p>2 → shape approaches square




Task 2. Implementation of a feature selection method based on entropy.
Generating a synthetic dataset of 5 features with different distributions and analyzing their entropy with different numbers of bins.
Validation is performed via Mutual Information (MI) from Scikit-Learn.

Step 1. Generating data with different entropy
//
    np.random.seed(42)
    n_samples = 500
//

As an example, we will give 5 features with different degrees of entropy

Step 2. We use Shannon's formula to calculate the entropy of features

//
def compute_entropy(data, bins=20):
    hist, _ = np.histogram(data, bins=bins, density=False)
    probabilities = hist / np.sum(hist)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))
//

Step 3. Estimating entropy for different bins
Calculating entropy for each feature with different numbers of bins.
//
bin_counts = [3, 10, 20, 50, 100]
entropy_results = {b: [compute_entropy(X[:, i], bins=b) for i in range(X.shape[1])] for b in bin_counts}
//

Step 4. Validation with Scikit-Learn
//
y = (features["Uniform"] > np.median(features["Uniform"])).astype(int)
mi_scores = mutual_info_classif(X, y, discrete_features=False)
//

Step 6. The average entropy is calculated for different bins.
//
average_entropy = {name: np.mean([entropy_results[b][i] for b in bin_counts]) for i, name in enumerate(feature_names)}
sorted_features = sorted(average_entropy.items(), key=lambda x: x[1], reverse=True)
//




Task 3. Unsupervised learning: density based methods.

Step 1. Implementation of the Generic Grid Clustering algorithm
//
class GenericGridClustering:
    def __init__(self, grid_size, density_threshold):
        self.grid_size = grid_size
        self.density_threshold = density_threshold
        self.grid = {}
//

Step 2. Define the boundaries of space and divide it into a grid.
        Count the number of dots in each cell.
        Filter the cells, leaving only dense areas.
//
def fit(self, X):
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    self.grid = {}
    for point in X:
        grid_x = int((point[0] - x_min) / self.grid_size)
        grid_y = int((point[1] - y_min) / self.grid_size)
        self.grid[(grid_x, grid_y)] = self.grid.get((grid_x, grid_y), 0) + 1
    self.clusters = {cell: count for cell, count in self.grid.items() if count >= self.density_threshold}
//

Step 3. Determining the belonging of new points in a cluster
//
def predict(self, X):
    labels = []
    for point in X:
        grid_x = int((point[0] - X.min(axis=0)[0]) / self.grid_size)
        grid_y = int((point[1] - X.min(axis=0)[1]) / self.grid_size)
        labels.append(1 if (grid_x, grid_y) in self.clusters else 0)
    return np.array(labels)
//

Step 4. Generate data
//
def generate_datasets():
    X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    return [(X_moons, 'Half Moons'), (X_blobs, 'Blobs')]
//

Step 5. Testing with different parameters
//
for X, name in generate_datasets():
    for grid_size in grid_sizes:
        for density_threshold in density_thresholds:
            print(f'Processing dataset: {name} (grid_size={grid_size}, density_threshold={density_threshold})')
            grid_cluster = GenericGridClustering(grid_size=grid_size, density_threshold=density_threshold)
            grid_cluster.fit(X)
            grid_cluster.plot_grid(X)
//



Task 4. Supervised learning

Step 1. Data generation and preparation
//
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
//

Step 2: Visualizing the Decision Boundary
//
def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=20)
    plt.title(title)
    plt.show()
    //

Step 3. Training the decision tree and testing with different depths (max_depth)
//
for max_depth in [1, 3, 5, None]:
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
//

The main thing is the pruning depth coefficients, it is important to avoid underfitting and overfitting to build a reliable model.
