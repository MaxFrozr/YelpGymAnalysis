import os
import plotly.express as px
import pandas as pd
import gensim

from load_business import load_business
from yelp_corpus import YelpReviewsCorpus

# load Gyms only
business_dataset_path = os.getenv('YELP_BUSINESS_PATH', 'yelp_academic_dataset_business.json.gz')

gyms_df = load_business(business_dataset_path, 'Gym')

# initialize the corpus to load only reviews about gyms with 1-star rating
reviews_dataset_path = os.getenv('YELP_REVIEWS_PATH', 'yelp_academic_dataset_review.json.gz')
corpus = YelpReviewsCorpus(reviews_dataset_path, gyms_df.business_id.values,
                           stars=1, custom_stopwords={'gym'})


# load corpus into memory. ideally corpus iterator can be passed directly to the LdaModel for memory efficiency.
# LdaModel iterates over the whole corpus several times, which takes too much time especially when file is hosted
# remotely This can be improved, but for relatively small corpus size it is OK.
buffered_corpus = []
for c in corpus:
    buffered_corpus.append(c)

print(f'Corpus size: {len(buffered_corpus)}')

# build Latent Dirichlet Allocation model
print("Building LDA model...")
lda_model = gensim.models.LdaModel(buffered_corpus, num_topics=10, id2word=corpus.dictionary,
                                   alpha='auto', chunksize=1000, random_state=1, passes=5)

# show list of topics
topics = lda_model.show_topics(formatted=False)
print(topics)

# convert topics to the dataframe of topic id, word and word weight
word_weigths = []
for i, topic in topics:
    for word, weight in topic:
        word_weigths.append((i, word, weight))

topics_df = pd.DataFrame(word_weigths, columns=('topic', 'word', 'weight'))

# visualize word weights for each topic
for i in range(len(topics)):
    df = topics_df[topics_df.topic == i].sort_values('weight')
    fig = px.bar(df, x='weight', y='word', orientation='h',
                 title=f'Topic {i + 1}', width=400, height=400)
    fig.write_image(f'output/topic_{i + 1}.png')
