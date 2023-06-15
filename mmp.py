import numpy as np
from sklearn.metrics import f1_score


class MaMiPot:
    def __init__(self, clf, sma=None, alfa_p=0.3, alfa_n=0.8, gamma=1, beta=0):
        '''
        Arguments:
        clf (scikit-learn classifier) : classification object
        sma (smote-variants oversampler) : sampler object
        alfa_p (float) : repositioning factor for positive samples
        alfa_n (float) : repositioning factor for negative samples
        gamma (int) : minimum number of true positive/negative samples for repositioning 
        beta (float) : proportion of the difference between majority and minority samples, decides how many
                       synthetic samples will be generated, works only if sm is not None
        '''

        self.clf = clf
        self.sma = sma
        self.alfa_p = alfa_p
        self.alfa_n = alfa_n
        self.gamma = gamma
        self.beta = beta

    # Calculates the centroid of a data subset
    def _get_centroid(self, data):
        if data.size > 0:
            return np.sum(data, axis=0)/data.shape[0]
        else:
            return 0
    
    def fit_resample(self, tr_data, tr_labels):
        labels, counts = np.unique(tr_labels, return_counts=True)
        minority_class = labels[np.argmin(counts)]
        n = (tr_labels != minority_class).sum() # number of negative samples
        p = (tr_labels == minority_class).sum() # number of positive samples

        self.clf.fit(tr_data, tr_labels) # train classifier
        pred_probs = self.clf.predict_proba(tr_data) 
        th_opt = (1/2)*(p/(p+n) + 0.5) # calculate optimal threshold for f1
        pred = (pred_probs[:,1] >= th_opt).astype(int)
        metric_opt = f1_score(tr_labels, pred)

        bool_array = np.logical_and(tr_labels != minority_class, pred != minority_class)
        tn_index = np.where(bool_array)[0]
        stn = np.take(tr_data, tn_index, axis=0) # set of true negatives
        stn_labels = np.take(tr_labels, tn_index, axis=0)
        
        bool_array = np.logical_and(tr_labels == minority_class, pred == minority_class)
        tp_index = np.where(bool_array)[0]
        stp = np.take(tr_data, tp_index, axis=0) # set of true positives
        stp_labels = np.take(tr_labels, tp_index, axis=0)
   
        bool_array = np.logical_and(tr_labels == minority_class, pred != minority_class)
        fn_index = np.where(bool_array)[0]
        sfn = np.take(tr_data, fn_index, axis=0) # set of false negatives
        sfn_labels = np.take(tr_labels, fn_index, axis=0)

        bool_array = np.logical_and(tr_labels != minority_class, pred == minority_class)
        fp_index = np.where(bool_array)[0]
        sfp = np.take(tr_data, fp_index, axis=0) # set of false positives
        sfp_labels = np.take(tr_labels, fp_index, axis=0)

        mu_p = self._get_centroid(stp)
        mu_n = self._get_centroid(stn)

        iter = 0
        tr_data_opt = np.copy(tr_data)
        tr_labels_opt = np.copy(tr_labels)

        while iter < 3:
            iter += 1

            if stp.shape[0] > self.gamma and sfn.shape[0] > 0:
                for i, x in enumerate(sfn):
                    # move false negative samples toward positive centroid
                    sfn[i] = (self.alfa_p * mu_p) + ((1 - self.alfa_p) * x)

            if stn.shape[0] > self.gamma and sfp.shape[0] > 0:
                for i, y in enumerate(sfp):
                    # move false positive samples toward negative centroid
                    sfp[i] = (self.alfa_n * mu_n) + ((1 - self.alfa_n) * y)

            # Create updated dataset
            tr_data_new = np.vstack((stp, stn, sfn, sfp))
            tr_labels_new = np.hstack((stp_labels, stn_labels, sfn_labels, sfp_labels))
            self.clf.fit(tr_data_new, tr_labels_new)
            pred_probs = self.clf.predict_proba(tr_data)
            pred = (pred_probs[:,1] >= th_opt).astype(int)
            metric_new = f1_score(tr_labels, pred)

            # Update the old dataset if new metric is highier
            if metric_new > metric_opt:
                metric_opt = metric_new
                tr_data_opt = tr_data_new
                tr_labels_opt = tr_labels_new
                iter = 0

        if self.beta > 0: # Oversample minority data
            self.sma.proportion = self.beta
            tr_data_res, tr_labels_res = self.sma.fit_resample(tr_data_opt, tr_labels_opt)
            return tr_data_res, tr_labels_res
        else:
            return tr_data_opt, tr_labels_opt
