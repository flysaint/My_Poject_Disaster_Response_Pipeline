import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function:Loads Data
    
    Input:
        messages_filepath(str): path of the messages data
        categories_filepath(str): path of the categories data
        
    Output:
        df_merge(obj): df merge with message and categories
    """
    
    df_categories = pd.read_csv(categories_filepath)
    df_messages = pd.read_csv(messages_filepath)
    df_merge = df_messages.merge(df_categories,on='id')
    
    return(df_merge)


def clean_data(df):
    """
    Function:Cleans data,including split the str into numeric features,and drop the duplicated data
    
    Input:
        df(obj): original data
        
    Output:
        df_clean(obj): The cleaned data come from category,and which been splited into numeric features
        
    """
    # 切分数据
    categories = df['categories'].str.split(";", expand=True)
    # 获取第一列
    row = categories.iloc[0,:]
    # 将第一列做切分，分成不同列名
    category_column_names = [col_cat.split('-')[0] for col_cat in row]
    # 
    categories.columns = category_column_names
    # 获取每个str 的 数字后缀
    for col in categories.columns:
        categories[col] = categories[col].apply(lambda row: row.split('-')[1])
    # 转换成 数字型
    for col in categories.columns:
        categories[col] = pd.to_numeric(categories[col])
    # 删除原始列
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df,categories], axis=1)
    # 删除重复数据
    df_clean = df[df.duplicated() == False]
    return df_clean


def save_data(df, database_filename,table_name='df_clean'):
    """
    Function:Stores  Data into SQLite database
    
    Input:
        df(obj):  DataFrame
        table_name(str): table name
        database_filename(str): The SQLite database which we save DataFrame
        
    Output:
        Nothing
    
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()