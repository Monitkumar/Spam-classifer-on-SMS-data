#Import Necessasry liraries
import pandas as pd

#Text Preprocessing libraries
import re 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

#Import data
spam_ham_data = pd.read_csv('SMSSpamCollection',sep = '\t',
                            names=['label','message'])

#Preprocessing script
corpus = []
for i in range(0,5572):
    review = re.sub('[^a-zA-Z]', ' ', spam_ham_data['message'][i])
    review = review.lower()
    review = review.split()
    #review = [ps.stem(word)for word in review if not word in stopwords.words('english')]
    review = [wnl.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    

#Create Vector Representation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer(max_features =  3000)
X = cv.fit_transform(corpus).toarray()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
spam_ham_data['le_encoded'] = le.fit_transform(spam_ham_data['label'])

#Model Building  
#1.Separate X and y
X = cv.fit_transform(corpus).toarray()
y = spam_ham_data['le_encoded']

#2.Train test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=12)
    
#Model Training
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train,y_train)
    
#Model Testing
y_pred = nb_classifier.predict(X_test)
    
#Model Evaluation
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
accuracy      = accuracy_score(y_test, y_pred) 
confusion_mat = confusion_matrix(y_test,y_pred)
precision = precision_score(y_test, y_pred)   
    