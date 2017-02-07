# wembed.jl
This packedge in inspired by [Deeplearning4j 0.4] (https://github.com/deeplearning4j/deeplearning4j). All the output values are compared with the same.

- Objective

Basic utils for googles word2vec model.

- Prerequisites

Download Google's pre-trained word2vec model [GoogleNews-vectors-negative300.bin.gz] (https://doc-0g-8s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/al13g6nqr9icmcn8qsbdkn4dh515mmqi/1486447200000/06848720943842814915/*/0B7XkCwpI5KDYNlNUTTlSS21pQmM?e=download) into wembed.jl/res/


- Installation

``` sh
cd .julia/v0.5/
git clone https://github.com/salonip/wembed.jl.git
```

- Usage

``` sh
using wembed

bin_path = "../res/GoogleNews-vectors-negative300.bin"
model_path = "../res/word2vec_model.hdf5"

wembed.bin2hdf5(bin_path, model_path) #converts the binary file of google word2vec to hdf5
wembed.init(model_path) # loads the hdf5 word2vec model to memory
wembed.getvector("queen", false) #returns the vector form of 300X1 of given word.Returns false if the word is not present
wembed.getsimilarity("king", "queen") #returns the similarity value of king and queen in float
wembed.wordnearest("king", 5) #returns the top 5 similar words of king with the relative distance

wordsAsString = "breakfast sushi dinner lunch"
wembed.odd1out(wordsAsString) #selects the odd one from the given words in wordsAsString

wordset1 = ["king", "queen"]
wordset2 = ["emperor", "empress"]
nsimilarity(wordset1, wordset2) #returns the similarity between the sets

wordset1 = ["king", "man"]
wordset2 = ["queen", "woman"]
nsimilarity(wordset1, wordset2)

# vectorize a sentence using vector of words in it where default value is 0.25 for unavailable words
sentence = "Julia is a developing language"
words = map(String, split(sentence))
vector = getvector(words, fill(0.25,300))

```

