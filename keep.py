from urllib2 import Request, urlopen, URLError

def get_response():
    request = Request("https://apiv2.bitcoinaverage.com/indices/global/ticker/BTCUSD")
    response = urlopen(request)
    return response

window_len = 50
predict_len = 10

table = list()

for i in range(0, window_len + predict_len):
	actual_response = get_response().read()
	actual_last = float(actual_response.split("last")[1].split("high")[0].replace('"','').\
					replace(':','').replace(',','').replace(' ','').replace('\n',''))
	actual_bid = float(actual_response.split("bid")[1].split("last")[0].replace('"','').\
					replace(':','').replace(',','').replace(' ','').replace('\n',''))
	actual_ask = float(actual_response.split("ask")[1].split("bid")[0].replace('"','').\
					replace(':','').replace(',','').replace(' ','').replace('\n',''))
	actual_volume = float(actual_response.split("volume")[1].split("changes")[0].replace('"','').\
					replace(':','').replace(',','').replace(' ','').replace('\n',''))
	table.append([actual_ask, actual_bid, actual_last, actual_volume])


print table
