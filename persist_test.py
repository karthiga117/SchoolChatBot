from services.embedding_service import EmbeddingService
import numpy as np
import faiss

print('creating service')
es = EmbeddingService()
print('index ntotal', es.index.ntotal)
# manually add a dummy vector without calling OpenAI
vec = np.random.rand(1, 1536).astype('float32')
faiss.normalize_L2(vec)
es.index.add(vec)
es.texts.append('dummy chunk')
print('added one dummy chunk, ntotal now', es.index.ntotal)
es.save()
print('saved, reloading new service')
es2 = EmbeddingService()
print('reloaded index ntotal', es2.index.ntotal)
