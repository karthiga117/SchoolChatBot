import requests
url='http://127.0.0.1:8000/upload-book'
files={'file':('MATH1.pdf', open(r'E:\PRANESH\school\book\MATH1.pdf','rb'),'application/pdf')}
data={'grade':'Grade8'}
resp=requests.post(url,files=files,data=data)
print(resp.status_code)
print(resp.text)
