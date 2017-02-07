module wembed
using HDF5
using Base.Collections

# core.jl
  export init, bin2hdf5, getsimilarity, getvector, wordnearest, odd1out, nsimilarity

  include("core.jl")
end # module
