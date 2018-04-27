from urllib2 import Request, urlopen, URLError
import time


while True:

	request = Request("https://apiv2.bitcoinaverage.com/indices/global/ticker/BTCUSD")

	response = urlopen(request)

	last = response.read().split("last")[1].split("high")[0].split(' ')[1].split(',')[0]

	print last

	


	time.sleep(1)