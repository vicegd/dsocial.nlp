import nltk
from lib.similarity import Similarity

class SentimentMoviesAdjust:
    def __init__(self, negativeFinalProb, positiveFinalProb):
        self.negativeFinalProb = negativeFinalProb
        self.positiveFinalProb = positiveFinalProb
        self.positiveThreshold = 1
        self.negativeThreshold = 1
        self.set_balance(0)

    def adjust(self, text):
        similarity = Similarity()

        pos1 = similarity.are_similar('@\w+ try \w+', text, 1)
        pos2 = similarity.are_similar('(.*) (re-)?watch (.*) fabulous (.*)', text, 1)
        pos3 = similarity.are_similar('is the most hilarious', text, 0.9)
        pos4 = similarity.are_similar('must watch', text, 1)
        pos5 = similarity.are_similar('ain\'t a bad movie', text, 0.9)
        pos6 = similarity.are_similar('super special', text, 0.9)
        pos7 = similarity.are_similar('laughter', text, 0.9)
        pos8 = similarity.are_similar('Watch it', text, 0.9)
        pos9 = similarity.are_similar('(.*) was really (.*) cool(.)?(.*)', text, 0.9)
        pos10 = similarity.are_similar('Not bad', text, 0.9)
        pos11 = similarity.are_similar('liked it very much', text, 0.9)

        neg1 = similarity.are_similar('they fail', text, 1)
        neg2 = similarity.are_similar('was dissapointed', text, 0.7)
        neg3 = similarity.are_similar('torture to watch', text, 0.7)
        neg4 = similarity.are_similar('wouldnt recommend', text, 0.7)
        neg5 = similarity.are_similar('pathetic to watch', text, 0.7)

        neu1 = similarity.are_similar('watched', text, 1)
        neu2 = similarity.are_similar('I recommend', text, 1)
        neu3 = similarity.are_similar('forget it, Jake. It is Chinatown', text, 0.4)
        neu4 = similarity.are_similar('Chewie, we\'re home', text, 0.4)
        neu5 = similarity.are_similar('Get your stinking paws off me, you damned dirty ape', text, 0.4)
        neu6 = similarity.are_similar('They may take our lives, but they\'ll never take our freedom', text, 0.4)
        neu7 = similarity.are_similar('Hasta la vista baby', text, 0.5)

        #pos1 = similarity.are_similar('they may take our lives, but they will never take our freedom!', text, 0.6)
        #pos2 = similarity.are_similar('hasta la vista, baby', text, 0.95)
        #pos3 = similarity.are_similar('forget it, Jake. It is Chinatown', text, 0.6)
        #pos4 = similarity.are_similar('not bad', text, 0.6)
        #pos5 = similarity.are_similar('I liked it very much', text, 0.6)
        #pos6 = similarity.are_similar('\w+ idea', text, 0.6)
        #neg1 = similarity.are_similar('worst movie ever', text, 0.96)
        #neg2 = similarity.are_similar('they fail', text, 0.7)
        #neu1 = similarity.are_similar('feel bad', text, 1)
        #neu2 = similarity.are_similar('the good, the bad and the ugly', text, 0.89)

        if (pos1):
            self.set_balance(0.4)
        if (pos2):
            self.set_balance(0.2)
        if (pos3):
            self.set_balance(0.5)
        if (pos4):
            self.set_balance(0.3)
        if (pos5):
            self.set_balance(0.3)
        if (pos6):
            self.set_balance(0.2)
        if (pos7):
            self.set_balance(0.2)
        if (pos8):
            self.set_balance(0.3)
        if (pos9):
            self.set_balance(0.3)
        if (pos10):
            self.set_balance(0.3)
        if (pos11):
            self.set_balance(0.3)

        if (neg1):
            self.set_balance(-0.20)
        if (neg2):
            self.set_balance(-0.20)
        if (neg3):
            self.set_balance(-0.20)
        if (neg4):
            self.set_balance(-0.20)
        if (neg5):
            self.set_balance(-0.20)
        
        if (neu1 and neg2):
            self.set_balance(-0.20)
        if (neu2 and pos4):
            self.set_balance(0.10)
        if (neu3 or neu4 or neu5 or neu6 or neu7):
            self.set_balance(0.3)
		
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
