# LT2222 V23 Assignment 3

# Part 1

<br>The logic of the script works like this:</br>
<p>First, the function *os.walk* is used in order to traverse the series subfolders of the inputed parent folder in order to read the email text files.</p>

<p>Then after the files are read, the authors and content are identified and extracted using a regex-based method. This is done by using a series of anchor points in the text (such as boilerplate words specific to the email formatting system) that
appear in the majority of emails.</p>

<p>After extracting the required information, it is then formatted into the shape of a dataframe for easy access. 
Afterwards, the authors' names are encoded by the following logic: the chosen author becomes "1" and the other authors become "2".</p>

<p>In order to properly encode the emails' content according to the specified dimensionality, the function *TfidfVectorizer* is used alongside its *max_features* option.</p>

<p>The data is split into a training and testing set in proportions specified by the user, the rows are marked through the use of the column "type" into "Test" and "Train" data and then merged back into a new dataframe, where the data is cleaned and prepped for being outputted into a .csv file with the name based on the rule: {author name} + "_table_vect.csv".</p>

<p>In order to run the feature extractor, the script must be run with this command:

 <p>python a3_features.py {path to dataset folder}/lt2222-v23/enron_sample/ "{chosen author name to be classified by}" {number of dimensions desired} </p>

 <p>as an example, one possible script call would look like this:
 python a3_features.py /scratch/lt2222-v23/enron_sample/ "Corman Shelley" 3</p>

 *Do note that it is needed to specify the name of it author in the format "LastName FirstName" 
</p>

# Part 2

<br>The logic of the script works like this:</br>

<p>-The dataset is read from the .csv file and split into training and testing datasets based on their *type* tag, which is then dropped.
-Subsequently, the X and Y training data are split off dinamically according to the file's number of columns, following the rule
that there is a single "authors" column for the labels and that the rest of the columns are the emails' content dimensions.
-After being sepparated, the 2 arrays are then processed by extracting their values and in the case of the Y array, flattening it.</p>

<p>Aftewards the 2 arrays are then merged together into a list of tuples of the shape (feature vector,label) in order to be assigned to a MyDataset object and loaded into a *dataloader* object to help with batching and feeding the data into the neural network.</p>

<p>The neural network itself is defined in the *Model* class as being a Perceptron model using LogSoftmax on its data before output. 
Its input layer is adjusted dynamically, based on the *dimensions* parameter of the model, in order to properly fit the dimensionality of the input data. The model itself is then used for defining the optimizer, thus granting it a learning rate of 0.003, the reasoning being that a small learning rate would prevent the model from converging too quickly.</p>

<p>The *loss* function is defined as being *CrossEntropyLoss*, as it computes the cross entropy loss between input logits and target and it's useful for training a classification problem with C classes like in this usecase.</p>

<p>The model uses 10 epochs as I found that out as being the optimum compromise between speed and eficacy, any more than that and the increases in accuracy were no longer significant.</p>

After training the model, the testing data is prepared and used for evaluating the model's accuracy. For that, a confusion matrix is used, which is calculated, printed to the console and plotted in order to have better visibility.

After testing the model extensively, my conclusions on its performance are mixed. When using the raw dataset, the model suffers from overfitting, as the ratios of "other authors" and "chosen author" are very unbalanced, the "other authors" making up more than 80% of the total dataset, including the entirety of the testing set. 

Of course, I continued to test and investigate after noticing this situation. I then manually modified the dataset in order to increase the proportion of the "other authors" to around 40%. This change had led to an apparent decrease in model accuracy reaching aproximately 68%, thus proving that it can at least detect the chosen author, but that it has a low sensitivity in detecting it. I still believe that there are numerous improvements which could be made both to the model and to the dataset, such as both increasing the in-model weight of the "chosen author" label and by employing better methods of data extraction.

# Part 3

For this part, I have added the ability to change the number of hidden layers, the size of the hidden layers and the option of choosing between 2 nonlinear functions: *nn.Tanh* and *nn.ReLU* but also the option of having no nonlinear functions by writing "None".

Running the script now requires this format: python a3_model.py {filename of .csv dataset} {number of hidden layers} {size of hidden layers} {nonlinear option}
As an example, the script can be run like this: python a3_model.py "Corman Shelley_table_vect.csv" 2 10 "Tanh"

After having tested it with both nonlinear functions and with different hidden layer numbers and sizes, I could not notice any change in the model's prediction accuracy. I believe that this is due to the dataset being used.


# Part 4

In regards to the ethical implications of the accessibility and usage of Enron's email data, a multitude of problems may arise concerning its usage, content, and transparency.

Firstly, I believe that the usage of personal data, no matter the legality of its provenance, in NLP projects can become a threat to one's privacy, as control over their data would be lost. Thus raising the need for adding a layer of anonymization over the data before its usage if it's not provided willingly and knowingly by users, such as masking certain elements with neutral labels, making individuals harder to be identified, and changing the word order.

This situation is already visible with Github's Copilot software, which uses as training data the entirety of Github's public repositories. This has the effect of not just plagiarizing other people's code, but also making visible their API keys and other personal information.

Secondly, the authors of many of those emails were regular employees of Enron, with no connection to its misdeeds, whose private information has been publicized against their will due to the company's legal troubles. This should not be excused by the existence of some individuals which were prosecuted due to the contents of their emails, as the judicial system should operate under the presumption of innocence. Thus, I consider the very source of the dataset as unethical as it's derived from a blanket decision meant to expose a select few suspect individuals, which ended up forfeiting the privacy of a significant amount of innocent employees.

Finally, there is the aspect of other people's sensitive information being discussed in the publicized mails. These third parties having never given their consent and many of them having next to no connection to Enron's illegal dealings. These third parties can range from friends and family all the way to businesses offering services to Enron such as catering, cleaning, or maintenance. Enron's subpoena has exposed not just Enron's internal emails but also the emails being replied to or forwarded that belonged to other individuals or organizations, who had no say in their publication and further use.

In conclusion, I would say that the given dataset has strong unethical aspects, which would discourage me from using them in other NLP projects without anonymizing the data first.

# Part A

For this I have rewrote the script such that the argument given as the number of hidden layers will be the upper limit of the hidden layer range for which the model will be ran. The precision and recall scores are calculated by the sklearn.metrics functions *precision_score* and *recall_score* and the precision-recall curve is plotted by matplotblib to the *precision_recall_curve.png* .

This script is run with this format: python a3_model.py {filename of .csv dataset} {number of hidden layers} {size of hidden layers} {nonlinear option}

As an example: python a3_model_PartBonusA.py "Corman Shelley_table_vect.csv" 5 15 "ReLU"