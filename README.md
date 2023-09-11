Hello dear user! 

This user manual will guide you through everything you need to know about the python files, their sources and outputs in our submission. Before we start, please check that in the same directory, there should be 5 .py files, 2 txt files and 2 directories in total. Without further adieu, let's start! 

Prepare: Place the DataCollection and RelevanceFeedback folders in the same directory as where we have the .py files. This is really important to let the program function smoothly!

Topics.txt: This is the source of the queries.

common-english-words.txt: This is the source of the queries.

stemming: This folder holds the stemming algorithm

Result: This folder stores the txt files of the output from the three models. Everytime you run the models the folder and files will be overlapped. 

coll.py: You do not need to run this. This is where we define the classes for document objects and document collection objects, and hold the function for parsing the main document files. 

bm25.py: Run this. You will get the output of bm25 weights for all of the documents in all of the datasets. The output will be stored in the Result folder after the code finishes running. If there is no Result folder in the first place, it will create one.

cosine_sim.py: Run this. You will get the output of cosine similarity values or all of the documents in all of the datasets. The output will be stored in the Result folder after the code finishes running. If there is no Result folder in the first place, it will create one.

Rocchio_cosine_sim.py: Run this. You will get the output of Rocchio cosine similarity values or all of the documents in all of the datasets. The output will be stored in the Result folder after the code finishes running. If there is no Result folder in the first place, it will create one.

Evaluation.py: Run this and it will print out the classification report for three models in the terminal. Please make sure that the Results folder is established and that there are all three outputs in the folder before you run the evaluation.py.

That’s all you need to know. Lest you run into any trouble, please shoot us an email at n10883789@qut.edu.au, we will do our best to assist you.

We hope you have fun with the codes!
