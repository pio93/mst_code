import random
import numpy as np
import warnings
from scipy.linalg import sqrtm
import smote_variants


class SWIM(smote_variants.base.OverSamplingSimplex):
    def __init__(self, proportion=1.0, alpha=0.25, random_state=None):
        super().__init__(random_state=random_state)

        '''
            Arguments:
            proportion (float) : proportion of the difference between majority and minority samples, decides how many
                                 synthetic samples will be generated
            alfa (float) :       spread around density contour
        '''

        self.proportion= proportion
        self.alpha= alpha

    @classmethod
    def parameter_combinations(cls):
        return [{'poroprtion': 1.00, 'alpha': 0.25},
                {'proportion': 0.75, 'alpha': 0.32},
                {'proportion': 0.50, 'alpha': 0.40},
                {'proportion': 0.25, 'alpha': 0.50}]
    
    def __init__(self, alpha=0.25, proportion=1.0):
        self.alpha = alpha
        self.proportion = proportion

    def _get_mean(self, data):
        if data.size > 0:
            return np.sum(data, axis=0)/data.shape[0]

    # get linearly independent columns by qr decomposition
    def _get_lin_independent_columns(self, matrix):
        _, r = np.linalg.qr(matrix)
        indices = set()
        for i, row in enumerate(r):
            arr = row[i:]
            index = np.where(arr > 0)[0]
            for ix in index : indices.add(ix)

        return indices

    def sample(self, tr_data, tr_labels):
        labels, counts = np.unique(tr_labels, return_counts=True)
        minority_class = labels[np.argmin(counts)]
        A = np.take(tr_data, np.where(tr_labels != minority_class)[0], axis=0)
        B = np.take(tr_data, np.where(tr_labels == minority_class)[0], axis=0)
        rate = A.shape[0] - B.shape[0]
        rate = int(rate*self.proportion)
        synthetic_data = []
        synthetic_labels = []

        rank = np.linalg.matrix_rank(A)
        dim = A.shape[1]
        columns = set()

        # If rank is highier than number of dimensions, get independent columns
        if rank < dim:
            columns = self._get_lin_dependent_columns(A)
            tr_data = tr_data[:, columns]

        mu_a = np.mean(A, axis=0)
        Ac = (A - mu_a)
        Bc = (B - mu_a)

        if np.linalg.det(np.cov(Ac.T)) <= 0:
            warnings.warn("Singular Matrix")
            return tr_data, tr_labels
        
        sigma = sqrtm(np.linalg.inv(np.cov(Ac.T)))

        if np.any(np.iscomplex(sigma)) == True:
            warnings.warn("Complex numbers")
            return tr_data, tr_labels
        
        Bw = np.matmul(Bc, sigma)

        sd_arr = np.std(Bw, axis=0)

        for _ in range(0, rate):

            x = Bw[random.randint(0, Bw.shape[0] - 1)]

            s = np.array([random.uniform(x[i] - self.alpha * sd_arr[i], x[i] + self.alpha * sd_arr[i]) for i in range(tr_data.shape[1])])

            s_norm = s * (np.linalg.norm(x)/np.linalg.norm(s))
            s_new = np.matmul(np.linalg.inv(sigma), s_norm)

            synthetic_data.append(s_new)
            synthetic_labels.append(minority_class)

        synthetic_data = np.array(synthetic_data)
        synthetic_labels = np.array(synthetic_labels)
        
        new_data = np.vstack((tr_data, synthetic_data))
        new_labels = np.hstack((tr_labels, synthetic_labels))

        return new_data, new_labels

    def get_params(self):
        return {'proportion': self.proportion, 'alpha': self.alpha}





