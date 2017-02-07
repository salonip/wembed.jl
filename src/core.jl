#=______________________________________________________________________________
______________________________________________________________________________=#
function bin2hdf5(bin_path, out_path)
  try
    word_vectors = readgooglebin(bin_path)
    model_keys=collect(keys(word_vectors))
    model_values=collect(values(word_vectors))
    info("writing model ... ")
    fid=h5open(out_path,"w")
    fid["keys"]= model_keys
    dset_label = d_create(fid, "vectors", datatype(Float32),
                 dataspace(1, size(model_values[1],1), size(model_keys, 1)))
    map(i -> dset_label[:,:,i] = model_values[i] ,collect(1:size(model_keys, 1)))
    close(fid)
  catch E
    warn("Failed: $E")
    return false
  end
  return true
end

#=______________________________________________________________________________
______________________________________________________________________________=#
function readgooglebin(bin_path)
  f=open(bin_path)
  word_vectors=Dict{String,Any}()
  info("reading bin file:")
  @time full = readbytes(f)
  i=13
  ncount = 0
  while i < size(full,1)
    per_complete = i/size(full,1) * 100
    ncount += 1
    ncount % 50000 == 0 ? info("progress: $(per_complete)%") : nothing
    stringbuf = ""
    while Char(full[i])!=' ' && Char(full[i])!='\n'
      stringbuf = "$stringbuf$(string(Char(full[i])))"
      i += 1
    end
    i += 1
    floatarray=reinterpret(Float32,full[i:(1200+i-1)])
    i += 1200
    floatarray = floatarray.*(1/(norm(floatarray)))
    word_vectors[stringbuf] = floatarray
  end
  close(f)
  return word_vectors
end

#=______________________________________________________________________________
______________________________________________________________________________=#
function init(model_path)
  try
    fid = h5open(model_path, "r")
    words = read(fid["keys"])
    vectors = read(fid["vectors"])
    close(fid)
    global vectors = vectors[1,:,:]
    global words_key = Dict(zip(words,1:size(words,1)))
    return true
  catch E
    warn("Failed: $E")
    return false
  end
end

#=______________________________________________________________________________
String,String-> getsimilarity(vec1::Array,vec2::Array)
similarity between two words
getsimilarity("king","queen")
______________________________________________________________________________=#
function getsimilarity(word1::String, word2::String)
  return getsimilarity(getvector(word1, false), getvector(word2, false))
end

#=______________________________________________________________________________
Array(Float32,1),Array(Float32,1)-> Float32
similarity between two vectors
getsimilarity(getvector("king"),getvector("queen"))
______________________________________________________________________________=#
function getsimilarity(vec1, vec2)
  return vec1 != false && vec2 != false ?
    (dot(vec1,vec2)/(norm(vec1)*norm(vec2))) : false
end

#=______________________________________________________________________________
ASCIIString-> Array(Float32,1)
vector representation of word
vec1 = getvector("king")
______________________________________________________________________________=#
function getvector(word::String, default_value)
  idx = get(words_key, word, false)
  return idx != false ? vectors[:,idx] : default_value
end

#=______________________________________________________________________________
______________________________________________________________________________=#
function  getvector(words, default_value)
  vectorset = Array(Float32, size(words, 1), 300)
  for i = 1:size(words,1)
    value = getvector(words[i], false)
    if value == false
      value = default_value
    end
    vectorset[i,:] = value
  end
  return vectorset
end

#=______________________________________________________________________________
Array,Int64,Array(ASCIIString,1)-> tuple
Nearest nTop words to the given list
wordNearest(wordvectors,nTop,words)
______________________________________________________________________________=#
function wordnearest(wordvectors::Array, nTop, words)
  pq = PriorityQueue(AbstractString,Float32)
  for word in setdiff(collect(keys(words_key)), words)
    dist = getsimilarity(wordvectors, vectors[:,get(words_key, word, false)])
    enqueue!(pq, word, dist)
    (length(pq) > nTop) && dequeue!(pq)
  end
  return sort(collect(pq), by = t -> t[2], rev=true)
end

#=______________________________________________________________________________
Array(String,1),Array(String,1),Int64-> wordnearest(wv::Array,nTop,words)
Nearest nTop words to the positve and negative list
wordnearest(wordvectors,nTop,words)
______________________________________________________________________________=#
function wordnearest(positive::Array, negative::Array, nTop)
  wordvectors = zeros(300)
  words = union(positive, negative)
  for word in words
    vec = getvector(word, false)
    if vec != false
      wordvectors += word in positive ? vec : vec.*(-1)
    else
      warn("$word does not exist")
      return false
    end
  end
  return wordnearest(wordvectors, nTop, words)
end

#=______________________________________________________________________________
ASCIIString,Int64-> tuple
Nearest nTop words to the positve and negative list
wordNearest(word,nTop)
______________________________________________________________________________=#
function wordnearest(word::String, nTop)
  gc()
  collection = wordnearest([word], [], nTop)
  return collection
end

#=______________________________________________________________________________
ASCIIString-> ASCIIString
odd word out from string with words separated by space
odd_one_out("breakfast sushi dinner lunch")
______________________________________________________________________________=#
function odd1out(wordsAsString)
  words=split(wordsAsString)
  words = intersect(collect(keys(words_key)), words)
  if isempty(words)
    warn("Cannot select from empty wordList.")
    return false
  end
  oddvectors = Array(Float32,size(words,1),300)
  for i =1:size(words,1)
    oddvectors[i,:] = vectors[:,words_key[words[i]]]
  end
  meanvec = unitvec(vec(mean(oddvectors,1)),"l2")
  dists = dotmat(oddvectors, meanvec)
  return sort(collect(zip(words,dists)), by = t-> t[2])[1][1]
end

#=______________________________________________________________________________
______________________________________________________________________________=#
function unitvec(arr::Array,norm_type)
  return norm_type == "l1" ? arr./norm(arr,1) : arr./norm(arr)
end

#=______________________________________________________________________________
______________________________________________________________________________=#
function dotmat(x::Array,y::Array)
  ans = Array(Float32,size(x,1))
  ans = map(i -> (ans[i] = dot(vec(x[i,:]),vec(y))),collect(1:size(x,1)))
  return vec(ans)
end

#=______________________________________________________________________________
list,list-> Float
Compute cosine similarity between two sets of words.
n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
______________________________________________________________________________=#
function nsimilarity(wordset1, wordset2)
  vectorset1 = getvector(wordset1, false)
  vectorset2 = getvector(wordset2, false)
  return vectorset1 != false && vectorset2 != false ?
     dot(unitvec(vec(mean(vectorset1,1)),"l2"),
     unitvec(vec(mean(vectorset2,1)),"l2")) : false
end
