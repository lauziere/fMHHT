
from config import *
from util import *

def main():
	
	for key in config:
	    print(key, ':', config[key])
	print('\n')
	
	tracker = fMHHT(config)

	tracker.track()

if __name__ == '__main__':
	
	main()

