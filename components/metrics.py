def length_metric(example, pred):
    """
    Check if the tweet is below the 280 character limit.
    Returns 1 if the tweet is below the limit, 0 if it is not.
    """
    return 1 if len(pred.tweet) <= 280 else 0

def hashtag_count_metric(example, pred):
    """
    Count hashtags in the tweet.
    Returns 1 if there are no hashtags, 0 if there are any hashtags.
    """
    tweet = pred.tweet
    hashtag_count = tweet.count('#')
    
    return 1 if hashtag_count == 0 else 0
