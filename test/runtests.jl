using wembed
using Base.Test

@test isfile("../res/GoogleNews-vectors-negative300.bin")

bin_path = "../res/GoogleNews-vectors-negative300.bin"
model_path = "../res/word2vec_model.hdf5"
@time @test wembed.bin2hdf5(bin_path, model_path)

@time @test wembed.init(model_path)
@time @test typeof(wembed.getvector("queen", false)) == Vector{Float32}
@time @test typeof(getsimilarity("king", "queen")) == Float32
@time @test typeof(wordnearest("king", 5)) == Array{Pair{AbstractString,Float32},1}

wordsAsString = "breakfast sushi dinner lunch"
@time @test typeof(odd1out(wordsAsString)) == String

wordset1 = ["king"]
wordset2 = ["man"]
@time @test typeof(nsimilarity(wordset1, wordset2)) == Float32
