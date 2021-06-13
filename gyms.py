import pandas as pd
import numpy as np
import plotly.express as px
from smart_open import open


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
        for chunk in reader:
            df = chunk[(chunk.categories.notnull()
                        & chunk.categories.str.contains(category)
                        & (chunk.is_open if only_opened else True))]
            df_list.append(df)
            counter += len(df)
            print(f'Number of {category} found: {counter}', end="\r")

    business_df = pd.concat(df_list)
    business_df = business_df.reset_index(drop=True)
    return business_df


# dataset file path
business_dataset_path = 'data/yelp_academic_dataset_business.json.gz'

# load Gyms only
gyms_df = load_business(business_dataset_path, 'Gym')

# group gyms by city, calculate total amount of gyms in each city and mean rating
gyms_by_city = gyms_df.groupby(['city', 'state']).agg(count=('name', 'count'), mean_stars=('stars', 'mean'))

# load US cities population dataset
cities_df = pd.read_csv('data/uscities.csv')


def get_city_population(df: pd.DataFrame, city: str, state: str) -> int:
    """
    Get population of the specified US city
    Args:
        df: dataframe with cities demographic information
        city: name of the city
        state: state of the city
    Returns:
        population of the city, NaN if city was not found in the dataset
    """
    pop = df.query('city == @city and state_id == @state')['population']
    return pop.iloc[0] if len(pop) > 0 else np.nan


# get population of each city with gyms
gyms_by_city['population'] = gyms_by_city.apply(lambda x: get_city_population(cities_df, x.name[0], x.name[1]), axis=1)

# calculate number of gyms per 100k inhabitants
gyms_by_city['rel_count'] = gyms_by_city['count'] / gyms_by_city['population'] * 1e5

# let's also filter cities by population and number of gyms, 
# assume that we are interested only in cities with > 50k inhabitants and > 5 working gyms
min_gyms_count = 5
min_population = 5e4
filtered_gyms = gyms_by_city.query('count >= @min_gyms_count and population >= @min_population')

# sort by relative number of gyms
filtered_gyms = filtered_gyms.sort_values('rel_count')

print('Top 10 cities with lowest number of gyms per 100k inhabitants')
print(filtered_gyms.head(10))

# scatter plot of the number of gyms per 100k inhabitants versus it's mean rating.
# Size of the bubble represent city population, color represents number of gyms in city
fig = px.scatter(filtered_gyms, x='rel_count', y='mean_stars', size='population', color='count',
                 title='Gyms across large US cities',
                 labels=dict(rel_count='Gyms per 100k inhabitants',
                             mean_stars='Mean gym rating',
                             count='Number of gyms in city',
                             text='City'),
                 text=filtered_gyms.index.values,
                 width=1000, height=500)

fig.write_image("output/gyms.png")
