import gensim.downloader as api

print("downloading glove-wiki-gigaword-50 ...")
model = api.load("glove-wiki-gigaword-50")
print("done, vocab size:", len(model.index_to_key))
model.save("C:/tmp/attention-experiment/glove50.kv")
