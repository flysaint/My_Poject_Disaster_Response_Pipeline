import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

'''
疑问1. GridSearchCV为什么使用demo里原始的参数，会报错。很可能是因为定义的结构不同导致
'''


def load_data(database_filepath,table_name = 'df_clean'):
    """
    Function:Loads disaster data from SQLite database
    
    Input:
        database_filepath(str): path of SQLite database
        table_name(str): name of the table where the data is stored
        
    Output:
        X(obj): array which contains the text messages
        Y(obj): array which contains the target of messages
        col_target(obj): The list with the target column names
    
    """
    
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name, engine)
    # 获取目标列
    targets = [col for col in df.columns if col 
                  not in ['id', 'message', 'original', 'genre','index']]
  
    X = df['message'] 
    Y = df[targets] 
    
    return X,Y, targets


def tokenize(text):
    """
    Function:Tokenizes a text
    
    Input:
        text(str): a raw text 
        
    Output:
        tokens(obj):The list of tokens based on the raw text input
    
    """
    text = re.sub("[^a-zA-Z0-9_]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return tokens

def build_model():
    """
    Function:Build a model including grid search
    
    Input:Nothing
        
    Output:
        cv(obj): an estimator which chains together a nlp-pipeline with
                    a multi-class classifier
    
    """
    
    # 初始化 梯度优化模型
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    
    ])    
    
    
    # 这里有疑问，为什么使用demo里原始的参数会报错。可能是因为定义pipeline的结构不同
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False)
        
     }
    
    
    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function:Evaluates the model performance for all categories on the test set
    
    Input:
        model(obj): a model used to pedict
        X_test(obj): an array with the test features
        Y_test(obj): an array which contains the targets (which correspond to X_test)
        category_names(obj): list with the category names
        
    Output:
        Nothing. But prints a classification report 
    
    """
    
    Y_test_df = pd.DataFrame(data = Y_test, columns=category_names)
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(data = Y_pred, columns=category_names)
    
    for cat_name in category_names:
        print("we are processing {} .... :".format(cat_name))
        print(classification_report(Y_pred_df[cat_name], Y_test_df[cat_name]))


def save_model(model, model_filepath):
    """
    Function:save the model into  a pickle file
        
    Input:
        model(obj): model for predict
        model_filepath(str): path to the location where the pickle file should be stored
        
    Output:
        Nothing
        
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()