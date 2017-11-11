
import requests

APIKEY = "HEYBMQFI2KUYRUUF"

DIGITAL_CURRENCY_MONTHLY = "DIGITAL_CURRENCY_MONTHLY"

BTC = "BTC"

CHINA = "CNY"

functions = {DIGITAL_CURRENCY_MONTHLY:"DIGITAL_CURRENCY_MONTHLY"}

crypto_symbols = {BTC:"BTC"}

ip="https://www.alphavantage.co/query?"


TIME_SERIES = "Time Series (Digital Currency Monthly)"



def get_digit_curr_monthly(market,coin):
	payload={"apikey":APIKEY}
	payload["function"] = functions[DIGITAL_CURRENCY_MONTHLY]
	payload["market"] = market
	payload["symbol"] = coin
	r = requests.get(ip,payload)
	
	json_monthly = r.json()
	print(json_monthly[TIME_SERIES])
	adap_data = {}
	for k in json_monthly[TIME_SERIES].keys():
		d = {"time":k,"val":json_monthly[TIME_SERIES][k]["1b. open (USD)"],"vol":json_monthly[TIME_SERIES][k]["5. volume"]}
		if k[:4] in adap_data.keys():
			adap_data[k[:4]].append(d)
		else:
			adap_data[k[:4]] = [d]

	return adap_data
