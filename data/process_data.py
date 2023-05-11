import sys
import pandas as pd 
import re
import sqlalchemy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, on = 'id', how = 'inner')
    
    return df 
    


def clean_data(df):
    categories = df['categories'].str.split(';', expand = True)


    # first row 
    row = categories.loc[0,:]

    category_columns = categories.loc[0,:].apply(lambda x: re.findall(r'.+?(?=-)', x)[0])

    categories.columns = category_columns
    
    for column in categories: 
        categories[column] = categories[column].apply(lambda x: re.findall(r'[0-9]', x)[0])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    
    df.drop(['categories'], axis = 1, inplace = True)
    
    
    df = pd.concat([df, categories], axis = 1)
    
    df = df.drop_duplicates()
    
    df.drop(['id', 'original'], axis = 1, inplace = True)
    
    return df



def save_data(df, database_filename): 
    
    engine = create_engine("sqlite:///" + database_filename)
    
    df.to_sql('disaster_data', engine, index = False, if_exists = "replace")
    
    return "Data added to database"


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