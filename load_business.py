import pandas as pd
from smart_open import open
from tqdm import tqdm


def load_business(filepath: str, category: str, only_opened: bool = True, chunksize: int = 1000):
    """
    Loads business data from yelp dataset, and filters it by specified category

    Args:
        filepath: path of the yelp dataset json file with business info
        category: name of the business category to select
        only_opened: if True, select only businesses that are opened
        chunksize: size of the chunk for file reading
    Returns:
        dataframe of selected businesses
    """
    df_list = []
    counter = 0
    with open(filepath) as f:
        reader = pd.read_json(f, orient='records', lines=True, chunksize=chunksize)
        for chunk in tqdm(reader, desc='Loading business data'):
            df = chunk[(chunk.categories.notnull()
                        & chunk.categories.str.contains(category)
                        & (chunk.is_open if only_opened else True))]
            df_list.append(df)
            counter += len(df)
            print(f" Number of {category}s found: {counter}", end="\r")

    business_df = pd.concat(df_list)
    business_df = business_df.reset_index(drop=True)
    return business_df
