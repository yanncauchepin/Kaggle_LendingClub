# Lending Club

Here is a repository related to the dataset [Lending Club](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1/code).

### Docker Image

You can access to my trained MLP binary classifier with the docker [image](https://hub.docker.com/repository/docker/yanncauchepin/lendingclub/general).

```sh
docker pull yanncauchepin/lending_club:mlp_binary_classifier
```

### How it works?

You can predict a the ***loan_status*** of a new data by using this model from two data insertion: manually or from json file. It depends on the choice you make in the menu of the docker image. To access to the menu, you must run the docker image in interactive mode -it.
```sh
docker run -it yanncauchepin/lending_club:mlp_binary_classifier
```

If you would like to use a json data, please ensure you provide data in the same format as the *data_test.json* file located in the **app** folder: make sure to write well the feature names. To bring the json data file in the docker image you must add a volume by running the following command, assuming you have a folder named **data** which contains your json data file *data.json* for example. Please update the *root_path* information according to your system.
```sh
docker run -it -v /root_path/data:/app/data yanncauchepin/lendingclub:mlp_binary_classifier
```
By choosing the second choice of predicting the ***loan_status*** from a data json file, you can get the output by providing *data/data.json* as input. If you inform *datat_test.json*, or simply press *Enter*, it will run the inner test.
