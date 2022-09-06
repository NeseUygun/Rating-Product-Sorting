
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales.
# The solution to this problem is to provide more customer satisfaction for the e-commerce site, to make the product stand out for 
# the sellers and It means a hassle-free shopping experience for buyers.
# Another problem is the correct ordering of the comments given to the products.
# Misleading product rating to come to the forefront will be cause the financial loss due to the fact that misleading comments
# will directly affect the sale of the product and will result in loss of customers.
# In the solution of these two basic problems, while the e-commerce site and the sellers increase their sales, the customers
# will complete the purchasing journey without any problems.

###################################################
# Dataset History
###################################################

# This dataset, which includes Amazon product data, includes product categories and various metadata.
# The product with the most reviews in the electronics category has user ratings and reviews.

# Variables:
# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)
# day_diff - Number of days since rating
# helpful_yes - The number of times the evaluation was found useful
# total_vote - Number of votes given to the evaluation


import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# TASK 1: Calculate average rating based on current comments and compare with existing average rating.
###################################################

###################################################
# Step 1: Read the dataset and calculate the average score of the product.
###################################################

df = pd.read_csv("datasets/amazon_review.csv")
df["overall"].mean()


###################################################
# Step 2: Calculate the weighted average score by date.
###################################################

# To calculate weighted points according to dates:
#   - assign reviewTime variable as date variable
#   - accept the max value of reviewTime as current_date
#   - create a new variable as day_diff by taking the difference of each point-comment date and current_date


# day_diff: How many days have passed since the comment was made?
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = pd.to_datetime(str(df['reviewTime'].max()))
df["day_diff"] = (current_date - df['reviewTime']).dt.days

# determination of time-based average weights
def time_based_weighted_average(dataframe, w1=24, w2=22, w3=20, w4=18, w5=16):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.2), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.2)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.4)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.4)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.6)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.6)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w4 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w5 / 100



###################################################
# Task 2: Specify 20 reviews for the product to be displayed on the product detail page
###################################################


###################################################
# Step 1. Create the helpful_no variable
###################################################


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

###################################################
# Step 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores and add to data
###################################################

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score 

    - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
    - The score to be calculated is used for product ranking.
    - Not:
    
    If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be made to conform to Bernoulli.
    This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

##################################################
# Step 3. Identify 20 interpretations and interpret the results.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)



