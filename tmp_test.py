from services.pdf_service import PdfService
p=PdfService()
with open(r'E:\PRANESH\school\book\SCIENCE.pdf','rb') as f:
    txt = p.extract_text(f.read())
print('len text', len(txt))
print(txt[:500])
from services.embedding_service import EmbeddingService
emb=EmbeddingService()
ids=emb.index_text(txt)
print('ids', ids)
print('chunks count', len(emb.texts))
