import numpy as np

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):


        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        root = DecisionTree(self.attribute_names.copy())
        root.tree = Tree()

        if (np.all(targets == 0)):
            root.tree.attribute_name = self.tree.attribute_name
            root.tree.value = 0
            return root
        elif(np.all(targets == 1)):
            root.tree.attribute_name = self.tree.attribute_name
            root.tree.value = 1
            return root
        if(len(self.attribute_names)== 0 or features.size == 0):
            root.tree.attribute_name = self.tree.attribute_name
            counts = np.bincount(targets.astype(int))
            root.tree.value = np.argmax(counts)
            return root
        else:
            IGArray = np.zeros( np.size(features, 1))
            for i in range(np.size(features,1)):
                IGArray[i] = information_gain(features, i, targets)

            index = np.argmax(IGArray)


            root.tree.attribute_index = index
            root.tree.attribute_name = root.attribute_names[index]
            attribute_column = features[:, index]



            examples1 = features[np.where(attribute_column < 1), :].squeeze(axis=0)
            targets1 = targets[np.where(attribute_column < 1)].squeeze()
            examples2 = features[np.where(attribute_column > 0), :].squeeze(axis = 0)
            targets2 = targets[np.where(attribute_column > 0)].squeeze()



            if(examples1.size == 0):
                counts = np.bincount(targets.astype(int))
                leaf = DecisionTree(root.attribute_names.copy())
                leaf.attribute_names.pop(index)
                leaf.tree = Tree()
                root.tree.value = np.argmax(counts)
                root.tree.branches.append(leaf)

            else:
                temp = examples1
                temp = np.delete(temp, index, 1)
                leaf = DecisionTree(root.attribute_names.copy())
                leaf.attribute_names.pop(index)
                leaf.tree = Tree()
                leaf = leaf.fit(temp, targets1)
                root.tree.branches.append(leaf)

            if(examples2.size==0):
                counts = np.bincount(targets.astype(int))
                leaf = DecisionTree(root.attribute_names.copy())
                leaf.attribute_names.pop(index)
                leaf.tree = Tree()
                leaf.tree.value = np.argmax(counts)
                root.tree.branches.append(leaf)
            else:
                temp = examples2
                temp = np.delete(temp, index, 1)
                leaf = DecisionTree(root.attribute_names.copy())
                leaf.attribute_names.pop(index)
                leaf.tree = Tree()
                leaf = leaf.fit(temp, targets2)
                root.tree.branches.append(leaf)
            return root




        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)

        raise NotImplementedError()

    def search(self, features):
        if(self.tree.value != None):
            return self.tree.value
        elif (self.tree.attribute_index != None and features[self.tree.attribute_index] == 0):
            features = np.delete(features, self.tree.attribute_index, axis = 0)
            return self.tree.branches[0].search(features)
        elif (self.tree.attribute_index != None and features[self.tree.attribute_index] == 1):
            features = np.delete(features, self.tree.attribute_index, axis = 0)
            return self.tree.branches[1].search(features)

    def predict(self, features):
        prediction = np.zeros(np.size(features, 0))
        for i in range(np.size(features, 0)):
            prediction[i] = self.search(features[i,:])
        return prediction








        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        self._check_input(features)

        #raise NotImplementedError()

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else -1
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        #print('dog')
        if not branch:
            branch = self.tree

        branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            branch.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    values = np.array([[0,0],[0,0]])

    example_size = np.size(features,0)
    for i in range(example_size):
        x = int(features[i, attribute_index])
        y = int(targets[i])
        values[x, y] = values[x,y] + 1

    total = np.sum(values)

    row0Sum = sum(values[0, :])
    row1Sum = sum(values[1, :])

    col0Sum = sum(values[:, 0])
    col1Sum = sum(values[:, 1])

    h  = -1* col0Sum/total * np.log2(col0Sum/total) - col1Sum/total * np.log2(col1Sum/total)

    if(values[0, 0] == 0 & values[0, 1] == 0):
        h1 = 0
    elif (values[0,0] == 0):
        h1 = row0Sum/total*(-1*values[0,1]/row0Sum*np.log2(values[0,1]/row0Sum))
    elif (values[0, 1] == 0):
        h1 = row0Sum / total * (-1 * values[0, 0] / row0Sum * np.log2(values[0, 0] / row0Sum))
    else:
        h1 = row0Sum/total*(-1 *values[0,0]/row0Sum *np.log2(values[0,0]/row0Sum) - 1*values[0,1]/row0Sum *np.log2(values[0,1]/row0Sum))

    if (values[1, 0] == 0 & values[1, 1] == 0):
        h2 = 0
    elif (values[1, 0] == 0):
        h2 = row1Sum / total * (-1 * values[1, 1] / row1Sum * np.log2(values[1, 1] / row1Sum))
    elif (values[1, 1] == 0):
        h2 = row1Sum / total * (-1 * values[1, 0] / row1Sum * np.log2(values[1, 0] / row1Sum))
    else:
        h2 = row1Sum/total*(-1 *values[1,0]/row1Sum *np.log2(values[1,0]/ row1Sum) - 1*values[1,1]/row1Sum *np.log2(values[1,1]/row1Sum))


    information_gain = h - (h1 + h2)
    return information_gain

   # h = total/example_size * ( ) np.sum(values)/np.size(features,0) *( -1*sum(values[0,:])/  sum(values)*np.log2())



'''

    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    

    raise NotImplementedError()
    
    '''


if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
