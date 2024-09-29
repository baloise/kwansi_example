import dspy

class TweetCreatorSignature(dspy.Signature):
    """Craft interesting tweets based on given topics and details. The tweets should be within the 280-character limit. Use a tone to suit the topic. Don't use hashtags."""
    
    topic = dspy.InputField(desc="the main subject of the tweet")
    details = dspy.InputField(desc="additional information or context for the tweet")

    tweet = dspy.OutputField(desc="a tweet")
