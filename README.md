# NLP-Assignment-2
The chosen case to be developed for this assignment is the first one, that is dividing a text in input into different enough slices such that they fit the context window of the Large Language Model of choice.

The selected LLM is **Llama-70b-Chat** from Meta whose context window consists in 2048 token.
 
 ## Structure
 The project is composed of 2 Python scripts:
 1. **Document.py** -> Python class used to represent the document, and to manage the slicing process. Slices are built up from sentences until they fix the context window, so that they are composed of full sentences. The slicing process is carried out through the following steps:

   i. if the document fits in the context window of the LLM, then it's the only slice to be returned
   
   ii. otherwise the text is divided into sentences
   
   iii. for each sentence:
     a. until the first slice hasn't been created, it is added to it
     b. if adding the sentence to the first slice makes it exceed the context window, the slice is added to the list, and we start creating the next slice from the second sentence of the first slice until this sentence that would've exceeded the context window
     c. add the sentence to the new slice to be created, and remove the first sentence on the list until the new slice is different enough (in terms of *cosine similarty* of the bag of words) from the first one
     d. add the sentence to the new slice to be created until it fits the context window
     e. if adding the sentence makes the new slice to be created exceed the context window, the slice is atted to the list, and we start creating the next slice in the same way as before

  To summarize the process, each new slice is a sort of window that scrolls down through the text starting from a slice, that goes from the second sentence of the last added slice to he list until the sentence right after the last one. This window scrolls down by removing sentences from the head, and appending new ones on the tail, until it's different enough from the last added slice, and fits the context window.

  The similarity between two slices is measured with **cosine similarity** on the bag of words of the slices, in which the freqquencies of the words are normalized depending on the length of the document. Two slices are considered to be similar if they're cosine similarity is greater than 0.8.

  In order to create the bag of words, the following pre-processing steps are taken on text:
    i. *punctuation removal*
    ii. *tokenization*
    iii. *stopwords removal*
    iv. *stemming*

 2. **main.py** -> The main script that feeds each slice to the LLM, and retrieves the answers.

## Dependencies
The following dependencies are used throughout the project:
- NLTK
- OpenAI: run the command
  ```shell
  pip install openai
  ```    
