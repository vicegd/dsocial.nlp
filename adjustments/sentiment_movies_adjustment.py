import nltk
from lib.similarity import Similarity

class SentimentMoviesAdjust:
    def __init__(self, negativeFinalProb, positiveFinalProb):
        self.negativeFinalProb = negativeFinalProb
        self.positiveFinalProb = positiveFinalProb
        self.positiveThreshold = 0.8
        self.negativeThreshold = 0.9
        self.set_balance(0)

    def adjust(self, text):
        similarity = Similarity()
        
        pos1 = similarity.are_similar('They may take our lives, but they will never take our freedom!', text, 0.6)
        pos2 = similarity.are_similar('Hasta la vista, baby', text, 0.95)
        neg1 = similarity.are_similar('worst movie ever', text, 0.96)
        neu1 = similarity.are_similar('Feel bad', text, 1)
        neu2 = similarity.are_similar('The Good, the Bad and the Ugly', text, 0.89)

        if (pos1):
            self.set_balance(0.15)
        if (pos2):
            self.set_balance(0.14)
        if (neg1):
            self.set_balance(-0.25)
        
        if ((pos1 and pos2)):
            self.set_balance(0.20)
        if (((neg1 and neu1) or (neg1 and neu2))):
            self.set_balance(-0.15)
		
        if (self.positiveFinalProb >= self.negativeFinalProb):
            result = 'positive'
        else:
            result = 'negative'
        return {'sentiment': result,
                'positive': self.positiveFinalProb,
                'negative': self.negativeFinalProb
        }

    def set_balance(self, value):
        if (value > 0):
            self.positiveFinalProb += value
            self.negativeFinalProb -= value
        else:
            self.positiveFinalProb -= -value
            self.negativeFinalProb += -value
        if (self.positiveFinalProb > self.positiveThreshold):
            self.positiveFinalProb = self.positiveThreshold
            self.negativeFinalProb = 1 - self.positiveThreshold
        if (self.negativeFinalProb > self.negativeThreshold):
            self.negativeFinalProb = self.negativeThreshold
            self.positiveFinalProb = 1 - self.negativeThreshold
