import numpy as np
import pickle as pck

#Track class
class Track(object):

    def __init__(self, track_id, feature_vector, genre):
        self._track_id = track_id
        self._feature_vector = feature_vector
        self._genre = genre

    @property
    def id(self):
        return self._track_id

    @property
    def feature_vector(self):
        return self._feature_vector

    @property
    def genre(self):
        return self._genre

    def num_word(self, index):
        return self._feature_vector[index]

    def num_words(self):
        return sum(self._feature_vector)

    def add_word(self, index):
        self._feature_vector[index] += 1

    def print_track(self):
        print('ID: %3d Genre: %7s' % (self.id, self.genre))
        print('Feature Vector: ' + str(self.feature_vector) + '\n')

#get features
def features():
    return np.load('data.npz')['arr_0'].tolist()

#get genres
def genres():
    return np.load('labels.npz')['arr_0'].tolist()

#get track id's
def track_ids(genre = None):
    tracks = pck.load(open('tracks.pck','r'))
    if genre == 'Rap':
        return tracks[:1000]
    elif genre == 'Pop_Rock':
        return tracks[1000:2000]
    elif genre == 'Country':
        return tracks[2000:3000]
    return tracks

#get dictionary
def get_dictionary():
    return pck.load(open('dictionary.pck','r'))

#get words
def get_words():
    words = np.load('words.npz')['arr_0']
    dictionary = get_dictionary()
    words = [dictionary[word] for word in words]
    return words

#get tracks
def get_tracks(genre = None):
    tracks = []
    for l,t,f in zip(track_ids(), features(), genres()):
        tracks.append(Track(l,t,f))
    if genre == 'Rap':
        return np.array([t for t in tracks if t.genre == 12])
    if genre == 'Pop_Rock':
        return np.array([t for t in tracks if t.genre == 1])
    if genre == 'Country':
        return np.array([t for t in tracks if t.genre == 3])
    return np.array(tracks)

#get vocabulary
def extract_vocabulary(D):
    V = []
    for word in words:
        V.append(word)
    return V

#get n documents
def count_documents(D):
    return len(D)

#get the n instances
def count_docs_in_class(D, c):
    return len([get_tracks(genre = c)])

#concatenate all of the text in docs
def concatenate_all_text_in_docs(D, c):
    text, tracks = [], get_tracks(genre = c)
    for word in words:
        for track in tracks:
            n = track.num_word(words.index(word))
            while n > 0:
                text.append(word)
                n = n - 1
    return text

#count instances of t in text_c
def count_tokens_of_terms(text_c, t):
    return text_c.count(t)

#extract all instaces of every word from doc, even if repeated
def extract_tokens_from_doc(V, d):
    text = []
    for word in words:
        n = d.num_word(words.index(word))
        while n > 0:
            text.append(word)
            n = n - 1
    return text

#train the multinomial model
def train_multinomial(C, D):
    V = extract_vocabulary(D)
    N = count_documents(D)
    prior, condprob = dict.fromkeys(C), {}
    for c in C:
        N_c = count_docs_in_class(D, c)
        prior[c] = float(N_c)/N
        text_c = concatenate_all_text_in_docs(D, c)
        T = {}
        for t in V:
            T[(t,c)] = count_tokens_of_terms(text_c, t)
        for t in V:
            condprob[(t,c)] = float(T[(t,c)] + 1)/(sum([T[tp,c] + 1 for tp in V]))
    return V, prior, condprob

#apply the multinomial on new instance d
def apply_multinomial(C, V, prior, condprob, d):
    W = extract_tokens_from_doc(V, d)
    score = dict.fromkeys(C, 0)
    for c in C:
        score[c] = np.log(prior[c])
        for t in W:
            score[c] += np.log(condprob[(t,c)])
    argmax, max_score = '', -np.inf
    for c in C:
        if score[c] > max_score:
            max_score = score[c]
            argmax = c
    return argmax

#run the multinomial model
def run():
    C, D = ['Rap', 'Pop_Rock', 'Country'], get_tracks()

    #applies the multinomial and prints the data
    def print_statistics(data):
        confusion_matrix = [[0,0,0,C[2]],[0,0,0,C[1]],[0,0,0,C[0]]]
        total = 0
        for d in [d for d in data if d.genre == tag['Rap']]:
            c = apply_multinomial(C, V, prior, condprob, d)
            if c == C[0]:
                total += 1
                confusion_matrix[2][0] += 1
            elif c == C[1]:
                confusion_matrix[2][1] += 1
            elif c == C[2]:
                confusion_matrix[2][2] += 1

        for d in [d for d in data if d.genre == tag['Pop_Rock']]:
            c = apply_multinomial(C, V, prior, condprob, d)
            if c == C[0]:
                confusion_matrix[1][0] += 1
            elif c == C[1]:
                total += 1
                confusion_matrix[1][1] += 1
            elif c == C[2]:
                confusion_matrix[1][2] += 1

        for d in [d for d in data if d.genre == tag['Country']]:
            c = apply_multinomial(C, V, prior, condprob, d)
            if c == C[0]:
                confusion_matrix[0][0] += 1
            elif c == C[1]:
                confusion_matrix[0][1] += 1
            elif c == C[2]:
                total += 1
                confusion_matrix[0][2] += 1

        print("classification accuracy: " + str(total*1.0/count_documents(data)))
        print('|%10s |%10s |%10s |' % (C[0],C[1],C[2]))
        print('-----------------------------------------------')
        for row in confusion_matrix:
            for col in row:
                print('|%10s' % col),
            print('')
        print('\n')

    #setup
    np.random.shuffle(D)
    V, prior, condprob = train_multinomial(C, D)

    # # uncomment if wanting to test on same data used to train
    # print_statistics(D)

    # # uncomment if wanting to print out the probabilities of a word given a genre
    # for c in C:
    #     for word in words:
    #         print('(%s,%s,%-.3f)' % (word, c, condprob[(word,c)]))

    # # uncomment if wanting to randomly generate song lyrics
    # def get_probabilistic_word(genre = None):
    #     assert(genre is not None)
    #     return np.random.choice(words, p = [condprob[(word, genre)] for word in words])
    # n_lyrics, n_songs = 20, 5
    # generated_tracks = []
    # for e, c in enumerate(C):
    #     for i in range(n_songs):
    #         t = Track(n_songs*e+i, [0]*30, tag[c])
    #         for j in range(n_lyrics):
    #             t.add_word(words.index(get_probabilistic_word(c)))
    #         generated_tracks.append(t)
    # for t in generated_tracks:
    #     t.print_track()
    # print_statistics(generated_tracks)

    # # uncomment if wanting to perform a k folds analysis
    # k = 10
    # for k_i in range(k):
    #     k_folds = np.split(D, k)
    #     test_data = k_folds.pop(k_i)
    #     train_data = [j for i in k_folds for j in i]
    #     V, prior, condprob = train_multinomial(C, train_data)
    #     print_statistics(test_data)

#start
tag = {'Rap' : 12, 'Pop_Rock' : 1, 'Country' : 3}
words = get_words()
tracks = get_tracks()
labels = genres()
run()
