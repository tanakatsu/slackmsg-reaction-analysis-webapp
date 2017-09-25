# Slack message sentiment analysis api


## API

- GET /predict
	- required parameter
		- `text`: input sentence (Japanese)

- POST /predict
	- required parameter
		- `text`: input sentence (Japanese)
	
##### Response sample
```
$ curl 127.0.0.1:5000/predict?text=おお、いいですね！ 

[["positive", 0.9580591917037964], ["negative", 0.04194079339504242]]
```
	
## Get started

Training scripts are available from [here](https://github.com/tanakatsu/chainer-sentiment-analysis).

### Prepare training dataset

Prepare a text file (`input_slack_messages.txt`: any name you like) that consists of slack message lines.

### Train dataset
1. Create input data and labels

	```
	$ python slack_message_dataset.py --mapfile reaction_map.yml input_slack_messages.txt
	```
	
    You can edit `reaction_map.yml` depends on your needs.
    
1. Run train.py

    ```
    $ python train.py --gpu 0 --input data/slack_comments.txt --label data/slack_labels.txt --seqlen 30 -e 20 -u 300
    ```
    
   You can configure some parameters.
   - --seq_len : sequence length
   - --unit : number of units
   - --e : epochs
     
1. Copy output files into `model/`
    - rnnlm.model
    - input\_vocab.bin
    - label\_vocab.bin

## Run on local machine

1. Clone this repository

	```
	$ git clone git@github.com:tanakatsu/slackmsg-reaction-analysis-webapp.git
	```

1. Copy trained model file and vocabulary files to `model/`
    - model/rnnlm.model
    - model/input\_vocab.bin
    - model/label\_vocab.bin
 
   [NB] Remember to edit `unit = 300` and `seq_len = 20` in `main.py` if you use your own parameters.
    
1. Build image

	```
	$ docker-compose build
	```

1. Start container

	```
	$ docker-compose up -d
	```
	
1. Navigate your browser to `$(docker-machine ip your-machine-name):5000/`


## Deploy to Heroku

1. Clone this repository

	```
	$ git clone git@github.com:tanakatsu/slackmsg-reaction-analysis-webapp.git
	```

1. Copy trained model files to `model/` and commit it.
    - rnnlm.model
    - input\_vocab.bin
    - label\_vocab.bin
   
   [NB] Remember to edit `unit = 300` and `seq_len = 20` in `main.py` if you use your own parameters.
       	
1. Create Heroku app

	```
	$ heroku create [your_app_name]
	```

1. Login to the registry
 
	```
	$ heroku container:login
	```

1. Build the image and push

	```
	$ heroku container:push web
	```

1. Open your web app in your browser

	```
	$ heroku open
	```
	
## License

MIT	