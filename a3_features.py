import os
import re
import string
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# hot encoding the author names based on the selected author function
def encode_author(df, author_name):
    df['authors'] = df['authors'].apply(lambda x: 1 if x == author_name else 2)
    return df


# text preprocessing function
def text_process(tex):

    # removing punctuation marks
    nopunct=[char for char in tex if char not in string.punctuation]
    nopunct=''.join(nopunct)
    return nopunct

# author and email content extractor function
def extract_author_and_content(email_string):
    list_s = []
    list_a = []

    # regex that extracts email author based on anchor words
    author_content = re.findall(r'(?<=From:\s)(?:(?!\.com)[^<\n])+ ', email_string)

    # regex that extracts email content based on anchor words
    email_content = re.findall(r'Subject:\t.*\n*.*\n?.*|Subject:\s.*\n.*\n.*\n?.*|pst\n*.*\n*.*', email_string)
    

    if email_content:
        
        # eliminating redundant and boilerplate characters and substrings
        email_content = [s.strip().replace("\n", "").replace("pst", "").replace("\t", "").replace("-----Original Message-----","").replace("RE:","").replace("Subject:","") for s in email_content]
        
        # removing redundant words from email content
        substring = "charset=us-ascii"
        substring2 = "Mime-Version"

        email_content = [elem for elem in email_content if substring not in elem]
        email_content = [elem for elem in email_content if substring2 not in elem]

        list_s.extend(email_content)

        # eliminating redundant and boilerplate characters
        author_content = [s.strip().replace("\t", "") for s in author_content]


        list_a.extend(author_content)

    return author_content, email_content


if __name__ == "__main__":

    # defining the script arguments
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("author", type=str, help="The name of the author to be classified.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))

    path = args.inputdir 


    authors = []
    texts = []

    # File parser for reading the data in the subfolder's files
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.'):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    author, content = extract_author_and_content(text)
                    authors.append([s.strip().replace("\t", "") for s in author])
                    texts.append(content)
  
    
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))

    columns = ['authors', 'text']
    df = pd.DataFrame(columns=columns)
    
    # creating the initial dataframe with the email's content and author
    for i in range(len(authors)):
        if len(texts[i]) > len(authors[i]):
            for y in range(len(authors[i])):
                new_row = {'authors': f'{authors[i][y]}', 'text': f'{texts[i][y]}'}
                df = df.append(new_row, ignore_index=True)
        else:
            for y in range(len(texts[i])):
                new_row = {'authors': f'{authors[i][y]}', 'text': f'{texts[i][y]}'}
                df = df.append(new_row, ignore_index=True)
    
    # preprocessing the authors' names
    df["authors"] = df.apply(lambda row: text_process(row.authors), axis = 1)
    
    # encoding the authors' names
    df = encode_author(df, args.author)
    df.dropna(subset=['text', 'authors'], inplace=True)

    # initialising vectoriser for inputed dimensions
    vectorizer_x = TfidfVectorizer(max_features=args.dims)

    X = vectorizer_x.fit_transform(df.text)

    Y = df.authors

    # splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.testsize/100, random_state=42)

    # convert the training data to pandas dataframe and write to file
    df_final_X = pd.DataFrame(X_train.toarray(), columns=vectorizer_x.get_feature_names())
    df_final_Y = pd.DataFrame(y_train, columns=["authors"])
    df_final = pd.concat([df_final_X, df_final_Y], axis=1)

    # marking the rows as training data
    df_final.insert(0, "type", "Train")

    # convert the testing data to pandas dataframe and write to file
    df_final_test_X = pd.DataFrame(X_test.toarray(), columns=vectorizer_x.get_feature_names())
    df_final_test_Y = pd.DataFrame(y_test, columns=["authors"])
    df_final_test = pd.concat([df_final_test_X, df_final_test_Y], axis=1)

    # marking the rows as training data
    df_final_test.insert(0, "type", "Test")

    # cleaning the dataframe and outputting it to .csv
    df_concat = pd.concat([df_final, df_final_test])
    df_concat = df_concat.fillna(0.0)
    df_concat = df_concat.drop(df_concat[df_concat['authors'] == 0.0].index)
    print(df_concat[df_concat['authors'] == 1])
    df_concat.to_csv(args.author + "_table_vect.csv", index=False)

    print("Done!")