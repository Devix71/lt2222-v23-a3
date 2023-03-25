import os
import sys
import glob
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def extract_author_and_content(email_string):
    list_s = []
    list_a = []

    author_content = re.findall(r'(?<=From:\s)(?:(?!\.com)[^<\n])+ ', email_string)
    email_content = re.findall(r'Subject:\t.*\n*.*\n?.*|Subject:\s.*\n.*\n.*\n?.*|pst\n*.*\n*.*', email_string)
    

    if email_content:
        
        email_content = [s.strip().replace("\n", "").replace("pst", "").replace("\t", "").replace("-----Original Message-----","") for s in email_content]
        
        
        substring = "charset=us-ascii"
        substring2 = "Mime-Version"

        

        email_content = [elem for elem in email_content if substring not in elem]
        email_content = [elem for elem in email_content if substring2 not in elem]
        print(email_content)
        list_s.extend(email_content)
        #print(list_s)
        author_content = [s.strip().replace("\t", "") for s in author_content]
        #print(author_content)

        list_a.extend(author_content)


        
    #print(len(list_s))
    #print(len(author_email))
    #email_content = email_string[content_start:] 

    return author_content, email_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("author", type=str, help="The name of the author to be classified.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    #print("Reading {}...".format(args.inputdir))
    path = '/home/devix/Documents/School/LT2222/Assignment3/lt2222-v23/enron_sample/panus-s/'
    #print(glob.glob(f'{path}*'))
    # Read the documents
    authors = []
    texts = []
    for file in glob.glob(f'{path}*'):
        #print(file)
        if os.path.isfile(file) and file.endswith('.'):
            with open(file, "r") as f:
                text = f.read()
                author, content =extract_author_and_content(text)
                
                authors.append( [s.strip().replace("\t", "") for s in author])
                texts.append(content)

    
    
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    

    print([len(s) for s in authors])

    print([len(x) for x in texts])

    columns = ['authors', 'text']
    df = pd.DataFrame(columns=columns)
    
    for i in range(len(texts)):
        for y in range(len(texts[i])):
            new_row = {'authors': f'{authors[i][y]}', 'text': f'{texts[i][y]}'}
            df = df.append(new_row, ignore_index=True)

    print(df)

    # Vectorize the documents
    vectorizer = TfidfVectorizer(max_features=args.dims)
    X = vectorizer.fit_transform(df.text)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df.authors, test_size=args.testsize/100, random_state=42)
    
    # Convert to pandas dataframe and write to file
    df_final = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names())
    df_final["author"] = y_train
    df_final.to_csv(args.author + "_table.csv", index=False)
    
    print("Done!")



    #python a3_features.py /home/devix/Documents/School/LT2222/Assignment3/lt2222-v23/enron_sample/panus-s/ tester 3