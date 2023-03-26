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

    print("Reading {}...".format(args.inputdir))
    if args.author == "all":
        path = args.inputdir
    else:
        path =  args.inputdir + args.author +"/"       #'/home/devix/Documents/School/LT2222/Assignment3/lt2222-v23/enron_sample/panus-s/'
    #print(glob.glob(f'{path}*'))
    # Read the documents
    authors = []
    texts = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.'):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    author, content = extract_author_and_content(text)
                    authors.append([s.strip().replace("\t", "") for s in author])
                    texts.append(content)
  
    
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    

    print([len(s) for s in authors])

    print([len(x) for x in texts])

    columns = ['authors', 'text']
    df = pd.DataFrame(columns=columns)
    
    for i in range(len(authors)):
        if len(texts[i]) > len(authors[i]):
            for y in range(len(authors[i])):
                new_row = {'authors': f'{authors[i][y]}', 'text': f'{texts[i][y]}'}
                df = df.append(new_row, ignore_index=True)
        else:
            for y in range(len(texts[i])):
                new_row = {'authors': f'{authors[i][y]}', 'text': f'{texts[i][y]}'}
                df = df.append(new_row, ignore_index=True)


 
    #print(df)

    # Vectorize the documents
    
    vectorizer_x = TfidfVectorizer(max_features=args.dims)
    vectorizer_y = TfidfVectorizer(max_features=1)

    X = vectorizer_x.fit_transform(df.text)
    Y = vectorizer_y.fit_transform(df.authors)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.testsize/100, random_state=42)
    #print(y_test.reset_index(drop=True))
    # Convert to pandas dataframe and write to file
    df_final_X = pd.DataFrame(X_train.toarray(), columns=vectorizer_x.get_feature_names())
    df_final_Y = pd.DataFrame(y_train.toarray(), columns=vectorizer_y.get_feature_names())
    df_final = pd.concat([df_final_X, df_final_Y], axis=1)
    df_final.insert(0, "type", "Train")
    #df_final.to_csv(args.author + "_table_train.csv", index=False)

    df_final_test_X = pd.DataFrame(X_test.toarray(), columns=vectorizer_x.get_feature_names())
    df_final_test_Y = pd.DataFrame(y_test.toarray(), columns=vectorizer_y.get_feature_names())
    df_final_test = pd.concat([df_final_test_X, df_final_test_Y], axis=1)
    df_final_test.insert(0, "type", "Test")
    #df_final_test.to_csv(args.author + "_table_test.csv", index=False)
    df_concat = pd.concat([df_final, df_final_test])
    df_concat.to_csv(args.author + "_table_vect.csv", index=False)

    print("Done!")



    #python a3_features.py /home/devix/Documents/School/LT2222/Assignment3/lt2222-v23/enron_sample/panus-s/ tester 3