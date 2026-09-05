from services.pdf_service import PdfService
p=PdfService()
with open(r'E:\PRANESH\school\book\SCIENCE.pdf','rb') as f:
    txt = p.extract_text(f.read())
print(repr(txt))
print('length', len(txt))
