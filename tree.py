import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    result = 0
    for elem in np.mean(y, axis = 0):
        result -= elem * np.log(elem + EPS)
    return result
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    #print(np.shape(y))
    #result = (np.bincount(np.array([int(elem) for elem in y]))) / np.size(y)
    result = np.mean(y, axis = 0)
    #print("gini probability array:")
    #print(result)
    result = 1. - np.sum(result**2)
    #print("gini value:")
    #print(result)

    return result
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    result = np.mean((y - y.mean())**2)
    
    return result

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    
    return np.sum(abs(y - np.median(y))) / np.size(y)


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        # YOUR CODE HERE
        #X_left  = np.zeros_like(X_subset)
        #X_right = np.zeros_like(X_subset)
        #y_left  = np.zeros_like(y_subset)
        #y_right = np.zeros_like(y_subset)
        #left_counter  = -1
        #right_counter = -1
        #for target_feature in X_subset[:, feature_index]:
        #    if target_feature < threshold:
        #        left_counter += 1
        #        X_left[left_counter, :] = X_subset[left_counter + right_counter + 1, :]
        #        y_left[left_counter, :] = y_subset[left_counter + right_counter + 1, :]
        #    else:
        #        right_counter += 1
        #        X_right[right_counter, :] = X_subset[left_counter + right_counter + 1, :]
        #        y_right[right_counter, :] = y_subset[left_counter + right_counter + 1, :]
        #X_left = X_left[0:left_counter + 1, :] if left_counter != -1 else np.array([], dtype = np.float64)
        #y_left = y_left[0:left_counter + 1, :] if left_counter != -1 else np.array([], dtype = np.float64)
#
        #X_right = X_right[0:right_counter + 1, :] if right_counter != -1 else np.array([], dtype = np.float64)
        #y_right = y_right[0:right_counter + 1, :] if right_counter != -1 else np.array([], dtype = np.float64)

        left_mask = X_subset[:, feature_index] < threshold
        right_mask = ~left_mask

        X_left,  y_left  = X_subset[left_mask],  y_subset[left_mask]
        X_right, y_right = X_subset[right_mask], y_subset[right_mask]


        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        #y_left  = np.zeros_like(y_subset)
        #y_right = np.zeros_like(y_subset)
        #left_counter  = -1
        #right_counter = -1
        #for target_feature in X_subset[:, feature_index]:
        #    if target_feature < threshold:
        #        left_counter += 1
        #        y_left[left_counter, :] = y_subset[left_counter + right_counter + 1, :]
        #    else:
        #        right_counter += 1
        #        y_right[right_counter, :] = y_subset[left_counter + right_counter + 1, :]
        #y_left  = y_left [0:left_counter  + 1, :] if left_counter  != -1 else np.array([], dtype = np.float64)
        #y_right = y_right[0:right_counter + 1, :] if right_counter != -1 else np.array([], dtype = np.float64)

        left_mask = X_subset[:, feature_index] < threshold
        right_mask = ~left_mask

        y_left, y_right  = y_subset[left_mask],  y_subset[right_mask]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        """
        functional_minimum = np.inf
        feature_index = -1
        threshold = -1
        current_index_of_feature = -1
        for current_feature in X_subset.T:
            current_index_of_feature += 1
            #print("y_size")
            #print(np.size(current_feature))
            functional4node = np.size(current_feature) * self.criterion(current_feature)
            for current_threshold in set(np.sort(current_feature)):
                y_left, y_right = self.make_split_only_y(
                    current_index_of_feature,
                    current_threshold,
                    X_subset, y_subset
                )
                #print("y_left size")
                #print(np.size(y_left) / 10)
                #print("y_right size")
                #print(np.size(y_right) / 10)
                functional = (functional4node - np.size(y_left / 10)  * self.criterion(y_left)
                                              - np.size(y_right / 10) * self.criterion(y_right))
                #print("functional")
                #print(functional)
                if functional < functional_minimum:
                    functional_minimum = functional
                    feature_index = current_index_of_feature
                    threshold = current_threshold
        #y_left, y_right = self.make_split_only_y(
        #            feature_index,
        #            threshold,
        #            X_subset, y_subset
        #        )
        #print("y_size")
        #print(np.size(current_feature))
        #print("functional_minimum:")
        #print(functional_minimum)
        #print("y_left - size:")
        #print(np.size(y_left) / 10)
        #print("y_right size")
        #print(np.size(y_right) / 10)
        """
    
        n_samples, n_features = X_subset.shape
        best_feature_index = None
        best_threshold = None
        best_impurity = float('inf')  # Initialize with a large impurity value

        # Loop over each feature to consider all possible splits
        for feature_index in range(n_features):
            # Get all unique values of the feature to use as candidate thresholds
            unique_values = np.unique(X_subset[:, feature_index])

            # Consider the midpoint between each pair of consecutive unique values as potential threshold
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                # Use make_split to split the data based on the current feature and threshold
                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)

                # Skip this split if it does not divide the dataset
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Compute the impurity for this split using weighted average of the impurity of left and right
                left_impurity = self.criterion(y_left)
                right_impurity = self.criterion(y_right)
                split_impurity = (len(y_left) / n_samples) * left_impurity + (len(y_right) / n_samples) * right_impurity
                #print(split_impurity)

                # Update best split if this one is better
                if split_impurity < best_impurity:
                    best_impurity = split_impurity
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        if np.size(X_subset, 0) >= self.min_samples_split and self.depth < self.max_depth:
            feature_index, treshold = self.choose_best_split(X_subset, y_subset)
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, treshold, X_subset, y_subset)
            new_node = Node(feature_index, treshold, proba = 0)
            if self.depth == 0:
                self.root = new_node
            self.depth += 1
            current_depth = self.depth
            #print("right")
            #print(current_depth)
            new_node.left_child  = self.make_tree(X_left,  y_left)
            
            self.depth = current_depth
            
            #print("left")
            #print(current_depth)
            new_node.right_child = self.make_tree(X_right, y_right)
            #if new_node.right_child == None and new_node.left_child == None:
            if not self.classification:
                new_node.proba = np.mean(y_subset, axis = 0)
            else:
                #print("list:")
                #print(current_depth)
                #print(np.bincount(np.array([int(elem) for elem in one_hot_decode(y_subset)])))

                #new_node.proba = np.argmax(np.bincount(np.array([int(elem) for elem in one_hot_decode(y_subset)])))
                new_node.proba = np.bincount(np.array([int(elem) for elem in one_hot_decode(y_subset)]))


                #print("proba:")
                #print(new_node.proba)
        else:
            new_node = None

        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        y_predicted = np.zeros([np.size(X, 0), 1])
        index = -1
        for object in X:
            index += 1
            leaf = self.root
            while leaf.left_child != None and leaf.right_child != None:
                if object[leaf.feature_index] < leaf.value:
                    leaf = leaf.left_child
                else:
                    leaf = leaf.right_child
            #y_predicted[index] = leaf.proba
            y_predicted[index] = np.argmax(leaf.proba) if self.classification else leaf.proba
        #print("y_predicted:")
        #print(y_predicted)
        
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        y_predicted_probs = np.zeros([np.size(X, 0), self.n_classes])
        index = -1
        for object in X:
            index += 1
            leaf = self.root
            while leaf.left_child and leaf.right_child:
                if object[leaf.feature_index] < leaf.value:
                    leaf = leaf.left_child if leaf.left_child else leaf
                else:
                    leaf = leaf.right_child if leaf.right_child else leaf
            y_predicted_probs[index, 0:len(leaf.proba)] = leaf.proba / np.sum(leaf.proba)
        
        return y_predicted_probs
