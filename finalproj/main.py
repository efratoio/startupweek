
# coding: utf-8


import data
import network
import sys, traceback
from os import listdir, path
from contextlib import redirect_stdout

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
    	self.log.close()


def evaluate_model(config,model,x_train, y_train,x_val, y_val,x_test,y_test):
	print(model.metrics_names)

	# if "RNN" in config.keys():
	# 	x_train = x_train[1]
	# 	x_val = x_val[1]
	# 	x_test = x_test[1]

	if (config["PROPS"] or config["SEMANTIC"]) and "RNN" not in config.keys():
		model.fit([x_train[0],x_train[1]], y_train, validation_data=([x_val[0],x_val[1]], y_val),epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"])
	else:
		model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"])


	
	if (config["PROPS"] or config["SEMANTIC"]) and "RNN" not in config.keys():
		return model.evaluate([x_test[0],x_test[1]],y_test, batch_size=config["BATCH_SIZE"], verbose=1, sample_weight=None)
	else:
		return model.evaluate(x_test,y_test, batch_size=config["BATCH_SIZE"], verbose=1, sample_weight=None)


def run_net(config,word_index,x_train, y_train,x_val, y_val,x_test,y_test):
	model = network.create_network(config,word_index)
	res = evaluate_model(config,model,x_train, y_train,x_val, y_val,x_test,y_test)
	print(str(res)+"\n")

def run_rnn(config,x_train, y_train,x_val, y_val,x_test,y_test):
	model = network.create_rnn(config)
	print(model)
	res = evaluate_model(config,model,x_train, y_train,x_val, y_val,x_test,y_test)
	print(str(res)+"\n")

  


  
def run_network(config):
	chats = data.gen_chat_data()
	word_index,x_train, y_train,x_val, y_val,x_test,y_test = data.prepare_datasets(config,chats)   
	if "RNN" in config.keys():
		run_rnn(config,x_train, y_train,x_val, y_val,x_test,y_test)
	else:
		run_net(config,word_index,x_train, y_train,x_val, y_val,x_test,y_test)




for file_name in listdir("./configurations"):
	with open(path.join("./experiments",file_name[:-5]+".log"),"w") as log_file:
		with redirect_stdout(log_file):
			config = data.load_config(path.join("./configurations",file_name))
			try:
				run_network(config)
			except:
				exc_type, exc_value, exc_traceback = sys.exc_info()
				traceback.print_exc(file=sys.stdout)
				print("Unexpected error: %s %s %s"%(exc_type, exc_value, exc_traceback))


