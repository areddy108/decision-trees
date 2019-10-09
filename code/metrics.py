import numpy as np

def confusion_matrix(actual, predictions):


    actual = 1*actual
    predictions = 1*predictions
    confusion_matrix = np.array([[0,0], [0,0]])
    for i in range(np.size(actual, 0)):
        confusion_matrix[int(actual[i]), int(predictions[i])] = confusion_matrix[int(actual[i]), int(predictions[i])]+1

    return confusion_matrix

    '''
    temp = confusion_matrix[0,0]
    confusion_matrix[0,0] = confusion_matrix[1,0]
    confusion_matrix[1,0] = temp
    '''



    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:



    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    #raise NotImplementedError()

def accuracy(actual, predictions):
    cm = confusion_matrix(actual, predictions)
    accuracy = (cm[0,0] + cm[1,1])/np.sum(cm)
    return accuracy





    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    #raise NotImplementedError()

def precision_and_recall(actual, predictions):
    cm = confusion_matrix(actual, predictions)
    precision = cm[1,1]/(cm[0,1] + cm[1,1])
    recall = cm[1,1]/(cm[1,0] + cm[1,1])
    return precision, recall

    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    #raise NotImplementedError()

def f1_measure(actual, predictions):

    _precision, _recall = precision_and_recall(actual, predictions)
    f1_measure = 2 * _precision*_recall/ (_precision + _recall)
    return f1_measure

    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    #raise NotImplementedError()

