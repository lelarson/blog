# chapter 1
My responses to the questionnaire in Chapter 1 of [Deep Learning for Coders with fastai and PyTorch by Jeremy Howard and Sylvain Gugger](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527).

1. Do you need these for deep learning?
	-  Lots of math? Nope
	* Lots of data? Not necessarily
	* Lots of expensive computers? No
	* a PhD? Definitely not

2. Name five areas where deep learning is now the best in the world.
			1. Computer vision
			2. Robotics
			3. Recommendation systems
			4. DNA sequencing
			5. Board games

3. What was the name of the first device that was based on the principle of the artificial neuron?
	* Mark 1 Perceptron
		* Frank Rosenblatt
		* Simulated on an IBM 704 computer at Cornell in 1957 
			* p. 193 in [Pattern Recognition and Machine Learning by Christopher M. Bishop](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)
	* Builds off the first documented mathematical model of an artificial neuron, the Threshold Logic Unit proposed by McCulloch and Pitts in 1943
		* The Mark 1 Perceptron built off this by :
			* Allowing any real inputs instead of binary values
			* Weighting different inputs differently
			* Adding a learning function
		* From [presentation by Dave Beeman at the University of Colorado](https://ecee.colorado.edu/~ecen4831/lectures/NNet2.html)
	
4. Based on the book of the same name, what are the requirements for Parallel Distributed Processing (PDP)?
	1. processing units
	2. a state of activation
	3. an output function for each unit
	4. a pattern of connectivity among units
	5. a propagation rule for propagating activities through the network
	6. an activation rule whereby patterns of connectivity are modified by experience
	7. environment in which the system will operate

5. What were the two theoretical misunderstandings that held back the field of neural networks?
	1. In 1969, a paper titled "Perceptrons" by Marvin Minsky and Seymour Papert was largely misinterpreted as showing that neural networks could only learn solutions to linear separable problems. In reality, the authors proved that such limitations only exist in the case of single-layer networks, like the Mark 1 Perceptron. However, they incorrectly speculated that this limitation might extend to more complex network models, and - despite being speculation - this appears to be the primary takeaway by the ML community given the subsequent decline in research funding for neural computing. Subsequently, the first AI winter began...
	2. ...only for neural networks to regain attention in the 1980s, when models with more than one layer were being explored. While it is possible to approximate any mathematical function using two layers of artificial neurons - and it was demonstrated that adding additional layers improved performance - this insight was not acknowledged, and these networks were too big and too slow to be of practical use anyway. Subsequently, the second AI winter began.

6. What is a GPU?
	- GPU = graphics processing unit (or graphics card)
	- Whereas a CPU is good for sequential processing, a GPU is better for simultaneously performing many mathematical calculations, such as those needed for deep learning. Because of this, the use of a GPU can dramatically speed up the training of a neural network model.

7. Open a notebook and execute a cell containing "1+1". What happens?
	-  We get 2.

8. Another interactive exercise
9. ... And another 

10. Why is it hard to use a traditional computer program to recognize images in a photo?
	- Like a baby who has not yet learned how to name the things it sees, a traditional computer program doesn't know what it doesn't know. There needs to be a process for it to identify different clues that could lead it to classify whatever its looking at as one thing or another. While a human child will have many different influences to help it learn efficient processes over time, a traditional computer program would need an explicit set of rules so it would know precisely what to look for. 
	- This is where machine learning comes into play. Instead of having inputs and a set of rules to get a desired results, we instead have our inputs and a set of ideal outputs, and the algorithm figures out the set of rules for us.
	
11. What did Samuel mean by “weight assignment”?
	- By "weights" he means variables where “weight assignment” refers to the current values of those variables
		- Because they will affect the program, they are in a sense another input using Samuel's usage

12. What term do we normally use in deep learning for what Samuel called “weights”?
	- parameters
	- Nowadays, "weights" refers to a particular type of model parameter
		- two *types* of neural network parameters are weights and biases

13. Draw a picture that summarizes Samuel’s view of a machine learning model.
![](chapter%201/Screen%20Shot%202021-03-03%20at%207.49.09%20PM.png)

14. Why is it hard to understand why a deep learning model makes a particular prediction?
	- referring to the interpretability of the model
	- Deep learning models are specifically hard to understand in part due to their “deep” nature. Unlike a linear regression model, where we can understand which variables are more or less important by their weights, deep neural networks have upwards of thousands of layers and, currently, there aren't many great ways to determine which factors are most important. 
		- However, there is progress in this area. For example, this chapter shows that we can analyze the sets of weights of a neural network model and determine what kind of features activate the neurons. More specifically, when applying CNNs to images, we can also see which parts of the images activate the model.
	
15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?
	- universal approximation theorem
		- due to imperfect data and the limitations of time and computer hardware, it is impossible to practically train a model to perfectly approximate all functions
		- but we can get damn close

16. What do you need in order to train a deep learning model?
	- data / measurements
		- for many use cases, labels for those data
	- an architecture for the network
	- a loss function to measure performance
	- a mechanism for updating the weights

17. How could a feedback loop impact the rollout of a predictive policing model?
	- a predictive policing model would be intended to predict crime
		- but the data may be measuring arrests, and any model using it would then be modeling arrests instead of crime
			- depending on existing police practices and any biases that exist in how people are arrested, the biases of these data would likely lead police to put more attention on areas with more arrests, leading to more arrests in those high-arrest areas
				- This is an example of a positive feedback loop, because the more the model is used, the more biased the data becomes, the more biased the model becomes

18. Do we always have to use 224×224-pixel images with the cat recognition model?
	- Not anymore, that is just the standard size for historical reasons 
		- old, pre-trained models required exactly 224x224 pixels
	- Increased picture size likely leads to higher accuracy since it will be able to focus on more details, but at the price of speed and memory consumption

19. What is the difference between classification and regression?
	- it's a difference in prediction: classification predicts a discrete class from available options while regression predicts a value from a continuous scale

20. What is a validation set? What is a test set? Why do we need them?
	- a validation test is the portion of data not used for training
		- used to validate the accuracy of the model during training and assess any overfitting
			- but there's a chance we overfit the validation data as well
				- because the human modeler is also apart of the process when picking hyper parameters
				- the test set is used for the final evaluation of the model
					- to ensure the model will generalize
	- if you don't have much data, a validation set may be enough

21. What will fastai do if you don’t provide a validation set?
	- automatically create one (can be specified with valid_pct = X)

22. Can we always use a random sample for a validation set? Why or why not?
	- Not always. While a random sample may be a good way of obtaining a validation set that is representative of your population, we always need to be mindful of the problem at hand. For example, when working with time series data, it would probably be cheating to make predictions about an earlier point in time, so our validation set would need to be the newest data you have

23. What is overfitting? Provide an example.
	- Overfitting is the issue of a model learning too much about the specific inputs you gave it instead of all possible inputs that it could be given for the problem at hand.
		- I.e., the model does not generalize to unseen data
		- especially important for neural networks because they can potentially memorize the dataset 

24. What is a metric? How does it differ from “loss”?
	- a metric is a function that measures the quality of the model's predictions using the validation set
		- Ex. error rate
	- this is similar to loss, which is also a measure of 
	- loss is meant to be used by the optimization mechanism while a metric is used by the human modeler

25. How can pre-trained models help?
	- pertained models are models that have already been trained on data for similar problems
	- useful in image detection because they have already learned 
	- when it works out, using them could save time/$ and you'd need less data
	- transfer learning refers to the usage of a pre-trained model
		
26. What is the “head” of a model?
	- When using a pre-trained model, the latter layers of that model are usually customized for the original problem. But we'll need to replace those for the new problem at hand. These new layers are referred to as the “head” of the model

27. What kinds of features do the early layers of a CNN find? How about the later layers?
	- Earlier layers learn simple features like diagonal, horizontal, and vertical edges
	- Latter layers learn more advanced features like car wheels, flower petals, and even outlines of animals

28. Are image models only useful for photos?
	- No, because many data can be creatively represented as images

29. What is an “architecture”?
	- refers to the kind of model we want to make
		- CNN vs RNN vs GAN vs others
	- only describes a template for a mathematical function
		- doesn't actually do anything until we provide values for the parameters it contains
		
30. What is segmentation?
	- pixelwise classification/labeling of what part of the picture each pixel represents

31. What is y_range used for? When do we need it?
	- used to limit the range of values that could be predicted
	- like when rating movies on a 1-10 scale
	
32. What are “hyper-parameters”?
	- parameters for our parameters
	- determine how our model is being trained
	- Ex: how long do we train for, what our learning rate is

33. What’s the best way to avoid failures when using AI in an organisation?
	1.  Make sure a training, validation, and testing set is defined properly in order to evaluate the model in an appropriate manner
	2. Try out a simple baseline, which future models should hopefully beat. Or even this simple baseline may be enough in some cases
	3. Make sure everyone agrees about what that point is
		1. that point should be your metric


