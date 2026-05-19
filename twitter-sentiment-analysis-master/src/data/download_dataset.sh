export KAGGLE_CONFIG_DIR="/tf/twitter-sentiment-analysis/kaggle"
if [ ! -f ../data/raw/training.1600000.processed.noemoticon.csv ];then	
	kaggle datasets download -d kazanova/sentiment140  -p ../data/raw
	unzip ../data/raw/*.zip -d ../data/raw
	rm ../data/raw/*.zip
fi
