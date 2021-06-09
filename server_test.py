import requests

url = "http://10.1.8.11:8005/ipopdf"

payload={'code': '9999',
'chi': 'IIS/2020/20200602/Attachment/HKEX-EPS_20200602_9302686_0.PDF',
'eng': 'IIS/2020/20200602/Attachment/HKEX-EPS_20200602_9302685_0.PDF',
'type': 'sponsors'}
files=[

]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)


