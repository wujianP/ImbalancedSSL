import json
import requests


def dingding_notifier(msg):
    token = "https://oapi.dingtalk.com/robot/send?access_token=0ad0909692b400e6fbdc64670f894eab7085424d87d143e34a312e998256113c"  # 这里替换为你刚才复制的内容
    headers = {'Content-Type': 'application/json'}
    data = {"msgtype": "text", "text": {"content": msg}}
    requests.post(token, data=json.dumps(data), headers=headers)


if __name__ == '__main__':
    dingding_notifier(keyword='New Best', msg="Top1-Acc:95.6\n @Epoch325\n ETA=1h32min")
