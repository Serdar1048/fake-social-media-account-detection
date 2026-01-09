from pandas.core.algorithms import value_counts
import pandas as pd

"""
pos | Number of total posts that the user has ever posted.
flg | Number of following
flr | Number of followers
bl  | Length (number of characters) of the user's biography
pic | Value 0 if the user has no profile picture, or 1 if has
lin | Value 0 if the user has no external URL, or 1 if has
cl  | The average number of character of captions in media
cz  | Percentage (0.0 to 1.0) of captions that has almost zero (<=3) length
ni  | Percentage (0.0 to 1.0) of non-image media. There are three types of media on an Instagram post, i.e. image, video, carousel
erl | Engagement rate (ER) is commonly defined as (num likes) divide by (num media) divide by (num followers)
erc | Similar to ER like, but it is for comments
lt  | Percentage (0.0 to 1.0) of posts tagged with location
hc  | Average number of hashtags used in a post
pr  | Average use of promotional keywords in hashtag, i.e.{regrann, contest, repost, giveaway, mention, share, give away, quiz}
fo  | Average use of followers hunter keywords in hashtag, i.e.{follow, like, folback, follback, f4f}
cs  | Average cosine similarity of between all pair of two posts a user has
pi  | Average interval between posts (in hours)
class | Fake (f) or Real (r)
"""
df = pd.read_csv("datasets/users.csv")

# Dataframe head
print(df.head())

# Dataframe shape is (65326, 18)
print(df.shape)

# Dataframe dtypes
print(df.dtypes)
"""
pos        int64
flw        int64
flg        int64
bl         int64
pic        int64
lin        int64
cl         int64
cz       float64
ni       float64
erl      float64
erc      float64
lt       float64
hc       float64
pr       float64
fo       float64
cs       float64
pi       float64
class     object
"""
# Dataframe null values: There is no null value in the dataset
print(df.isna().sum())