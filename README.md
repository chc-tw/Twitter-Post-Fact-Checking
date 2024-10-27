# Tweet Fact-Checking with BERT

## Project Overview
This project aims to fine-tune a pre-trained BERT model for fact-checking using a Twitter dataset. The model is trained to classify claims based on their veracity, leveraging the capabilities of BERT for natural language understanding.

## Contents of the Project
The project includes the following components:
- Data processing scripts to prepare the Twitter dataset.
- A model training script that utilizes the BERT architecture for sequence classification.
- Utility functions for reporting results and creating predictions.

## Environment Setup

To set up the environment for this project, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment**
   This project uses pipenv to manage dependencies. With provided Pipfile, you can install the dependencies by running:
   ```bash
   pip install pipenv
   pipenv install
   ```
   After installing the dependencies, you can activate the virtual environment by running:
   ```bash
   pipenv shell
   ```

3. **Download NLTK Resources**
   You may need to download additional NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Running the Code

To run the code, follow these steps:

1. **Prepare the Dataset**
   Ensure that your dataset is in the correct format (JSON) and located in the appropriate directory.

2. **Train the Model**
   Execute the training script:
   ```bash
   python main.py
   ```

3. **Evaluate the Model**
   After training, the model will automatically evaluate its performance on the validation set and save the best model.

4. **Make Predictions**
   The model will generate predictions for the test dataset and save the results in a specified output file.

## Expected Results

Upon successful execution, you can expect the following outputs:
- Training loss and accuracy metrics printed to the console during training.
- A confusion matrix and classification report summarizing the model's performance on the validation set.
- A JSON file containing the predictions for the test dataset, which can be used for further analysis or submission.

## Dataset

The training data for this project is structured in JSON format, containing pairs of text and their corresponding labels. Each entry consists of two texts (`text1` and `text2`) and a label indicating the veracity of the claim (0 for false, 1 for true). Below are some examples from the training dataset:

### Example Entries

1. **Entry 1**
   ```json
   {
       "text1": "['This is incredible, hilarious, and pathetic at the same time.', 'Theay re worst but this image is edited', 'This is some seriously insane shit 🤣', 'IS this real????', 'This timeline is where you’ll spend most of your time, getting instant', \"Ma'am appreciate 4G is back in Kashmir ☺️\", '1.   1. This Tweet is unavailable.', 'We’ve detected that JavaScript is disabled in this browser. Please enable', 'JavaScript is not available.']",
       "text2": "opindia claimed greta thunberg 's real name is ghazala bhat",
       "label": "0"
   }
   ```

2. **Entry 2**
   ```json
   {
       "text1": "['you remember that very well, during your term, during you and Barack', 'key promise during the administration. It also presided over record', 'out of that meeting — not from his administration, but from some of the', 'secondly, we’re in a situation here where the federal prison system was', '38,000 prisoners were released from federal prison, we have… There were', 'Reaction from readers in Moscow', 'by state mass media, and the attacks were related to the attempts to push', 'corruption, although they only did this once they were given a signal from', 'as the mayor of Moscow,\" Mr Medvedev told journalists during a visit to', 'harsh criticism from the Kremlin.']",
       "text2": "prisoner were released from federal prison during the obama administration.",
       "label": "1"
   }
   ```

3. **Entry 3**
   ```json
   {
       "text1": "['Issues of dispute\\xa0regarding the content of the leaks were legion.', 'US government, including the FBI and White House (who have reportedly', 'breach controversy during which the Sanders campaign was accused by the', 'users widely accused Facebook and Twitter of censoring the leaks), before', 'one—it’s not clear—of the hacking allegations that the DNC says have', 'Christians to the site of her birth.']",
       "text2": "a -year-old boy wa accused of hacking the fbi 's database.",
       "label": "0"
   }
   ```

### Data Format

- **text1**: A list of comments or statements related to a claim.
- **text2**: A specific claim that is being evaluated.
- **label**: An integer indicating the veracity of the claim (0 for false, 1 for true).

This dataset is used to train the BERT model to classify claims based on their veracity, helping to identify misinformation and support fact-checking efforts.

## Conclusion

This project demonstrates the application of BERT for fact-checking claims made on Twitter. By fine-tuning a pre-trained model, we can leverage its understanding of language to classify claims effectively. The results can be used to assess the reliability of information shared on social media platforms.
