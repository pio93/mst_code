import random
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import smote_variants

    
class SMOTEFUNA(smote_variants.base.OverSamplingSimplex):
    def __init__(self, proportion=1.0, random_state=None):
        super().__init__(random_state=random_state)

        '''
            Arguments:
            proportion (float) : proportion of the difference between majority and minority samples, decides how many
                                 synthetic samples will be generated
        '''


        self.proportion = proportion

    @classmethod
    def parameter_combinations(cls):
        return [{'proportion': 1.0}]
    
    # Get farthest neighbor
    def _get_max_dist_nghb(self, X, n):
        dist_vect = squareform(pdist(X, metric='cityblock'))[n]
        return X[np.argmax(dist_vect)]
    
    # Get distance to closest neighbour
    def _get_min_dist(self, X, s):
        dist_vect = cdist(X, s, metric='cityblock')
        return np.min(dist_vect)

    def sample(self, tr_data, tr_labels):
        labels, counts = np.unique(tr_labels, return_counts=True)
        minority_class = labels[np.argmin(counts)]
        majority = np.take(tr_data, np.where(tr_labels != minority_class)[0], axis=0)
        minority = np.take(tr_data, np.where(tr_labels == minority_class)[0], axis=0)
        rate = majority.shape[0] - minority.shape[0]
        rate = int(rate*self.proportion)
        i = 0
        synthetic_data = []
        synthetic_labels = []
        while i < rate:
            n = random.randint(0, minority.shape[0] - 1)
            s1 = minority[n]
            s2 = self._get_max_dist_nghb(minority, n)
            s_new = np.zeros(s1.shape[0])
            for j in range(s1.shape[0]):
                start = min(s1[j], s2[j])
                stop = max(s1[j], s2[j])
                s_new[j] = random.uniform(start, stop)
            theta = self._get_min_dist(minority, np.array([s_new]))
            beta = self._get_min_dist(majority, np.array([s_new]))
            # If closest neighbour is of minority class, add new sample
            if theta <= beta:
                synthetic_data.append(s_new)
                synthetic_labels.append(minority_class)
                i += 1
            
        synthetic_data = np.array(synthetic_data)
        synthetic_labels = np.array(synthetic_labels)

        new_data = np.vstack((tr_data, synthetic_data))
        new_labels = np.hstack((tr_labels, synthetic_labels))

        return new_data, new_labels

    def get_params(self):
        return {'proportion': self.proportion}
    

