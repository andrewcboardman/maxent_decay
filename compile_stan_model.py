import pystan
import argparse
import pickle

def main():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-if', '--infile', type=str, action='store', dest='infile',
	help='Name of file containing stan code')
	parser.add_argument('-of', '--outfile',type=str,action='store',dest='outfile',
	help='Name of file to write pickled model to')
	
	args = parser.parse_args()

	sm = pystan.StanModel(file=args.infile)

	with open(args.outfile,'wb') as fid:
		pickle.dump(sm,fid)

if __name__ == '__main__':
	main()