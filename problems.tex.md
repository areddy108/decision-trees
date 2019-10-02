# Coding (5 points)
Your task is to implement two machine learning algorithms:

1. Decision tree (in `code/decision_tree.py`)
2. Prior probability (in `code/prior_probability.py`)

You will also write code that reads in data into numpy arrays and code that manipulates
data for training and testing in `code/data.py`.

You will implement evaluation measures in `code/metrics.py`:

1. Confusion matrix (`code/metrics.py -> confusion_matrix`)
2. Precision and recall (`code/metrics.py -> precision_and_recall`)
3. F1-Measure (`code/metrics.py -> f1_measure`)

The entire workflow will be encapsulated in `code/experiment.py -> run`. The run function 
will allow you to run each approach on different datasets easily. You will have to 
implement this `run` function.

Your goal is to pass the test suite (contained in `tests/`). Once the tests are passed, you 
may move on to the next part - reporting your results.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here. Suggested order for passing test_cases:

1. test_load_data
2. test_train_test_split
3. test_confusion_matrix
4. test_accuracy
5. test_precision_and_recall
6. test_f1_measure
7. test_experiment_run_prior_probability
8. test_experiment_run_decision_tree
9. test_experiment_run_and_compare

# Free-response questions (5 points)

1. There is a difference between what a decision tree can represent and the learning algorithm used to generate a tree. Assume you have a deterministic function that takes a fixed, finite number of Boolean inputs and returns a Boolean output. Can a decision tree represent any such function? Give a simple proof for your answer. 

2. Let $v$  be a voter in the set of voters $V$. Let each $v$ have a value, assigned to either -1 or 1. Let's define Majority-rule as the sign of the sum all voters. If over half of the voters say -1, it returns -, if over half return 1, it returns +. Assume an odd number of voters (so there are no ties). Can the majority-rule algorithm be represented by a decision tree that considers a single voter at each decision node? Why or why not?

3. How well did your decision tree built using ID3 learn the majority-rule function? Describe the tree depth and the accuracy. How did this learning compare to how well it did on the Ivy League problem? 

4. What about each of these data sets made the ID3 learning algorithm suitable or unsuitable for learning the function for each data set?

5. Describe general features of an underlying decision problem (as opposed to the particular training set used to teach a decision tree) that will cause a decision tree like the ones you coded up to work well.

6. One can modify the simple ID3 algorithm to handle attributes that are real-valued (e.g. height, weight, age). To do this, one must pick a split point for each attribute (e.g. height > 3) and then determine information gain, given the split point. How would you pick a split point automatically? 

7. Describe an example data distribution for which your approach for creating split points would not work well. Describe a data distribution where it would work well. Give your reasoning.

8. Assume each person in a population is has two real-valued measured attributes: height, weight. In a two-dimensional plot, illustrate the decision line for the concept $height > weight$.

9. Assume you have a decision tree that uses a single real-valued attribute (plus an ideally-chosen split point value) at each decision node. Can you represent the concept $height > weight$ with such a tree?  In other words, you want to return "true" if someone's weight is greater than their age and "false" otherwise.  Can you do it? If so, specify the decision tree. If not, say why not.

10. In a 2-D plot, similar to that for question 8, plot the decision surface produced by the best decision tree you found in answering question 9.


