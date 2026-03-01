from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
analyzer = SentimentIntensityAnalyzer()

class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/docs", response_model=SentimentResponse)
async def analyze_comment(data: CommentRequest):
    try:
        scores = analyzer.polarity_scores(data.comment)
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "positive"
            rating = min(5, round(3 + compound * 2))
        elif compound <= -0.05:
            sentiment = "negative"
            rating = max(1, round(3 + compound * 2))
        else:
            sentiment = "neutral"
            rating = 3

        return SentimentResponse(sentiment=sentiment, rating=rating)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))