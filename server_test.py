import requests,json
from typing import Dict,List,Union,Tuple
from Config.setting import config


#url = "http://10.1.8.11:8005/ipopdf"
url = "http://127.0.0.1:8005/ipopdf"
# # payload={'code': '9999',
# # 'chi': 'IIS/2020/20200602/Attachment/HKEX-EPS_20200602_9302686_0.PDF',
# # 'eng': 'IIS/2020/20200602/Attachment/HKEX-EPS_20200602_9302685_0.PDF',
# # 'type': 'sponsors'}

payload={'code': '2175', 'chi': 'IIS/2021/20210630/Attachment/HKEX-EPS_20210630_9822171_0.PDF', 'eng': 'IIS/2021/20210630/Attachment/HKEX-EPS_20210630_9822170_0.PDF',
'type':'sponsors',}
files=[]
headers = {}

#response = requests.request("POST", url, headers=headers, data=payload , files=files)
response = requests.request("POST", url, headers=headers, data=json.dumps(payload), files=files)
#response=response.json()
response=json.loads(response.json())
print(response)

#
# def get_secfirm_list(sec_firm_api:str)->Dict:
#     resp=requests.get(sec_firm_api)
#     if resp.status_code is not 200:
#         return None
#     else:
#         return {row.get('cSponsorNameEng'):{'chi':row.get('cSponsorNameChi'),'id':row.get('cSponsorID')} for row in json.loads(resp.content)['content']}
#
#
# secfirm_list=get_secfirm_list(sec_firm_api='http://10.3.1.214/backend/api/get_ipo_sponsors.php')
# for eng in secfirm_list:
#     chi=secfirm_list.get(eng).get('chi')
#     with open('firm.txt','a',encoding='utf-8') as f:
#         f.write(eng+'\n')