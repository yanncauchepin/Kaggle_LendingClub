# Lending Club

Here is a repository related to the **dataset** [Lending Club](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1/code).

Associated **notebook** is also pusblished [here](https://www.kaggle.com/code/yanncauchepin/semi-supervised-learning-mlp-binary-classifier). 

### Docker Image

You can access to my trained MLP binary classifier with the **docker** [image](https://hub.docker.com/repository/docker/yanncauchepin/kaggle_lendingclub/general).

```sh
docker pull yanncauchepin/lending_club:mlp_binary_classifier
```

### How it works?

You can predict the ***loan_status*** of a new data by using this model from two data insertion: manually or from json file. It depends on the choice you make in the menu of the docker image. To access to the menu, you must run the docker image in interactive mode -it.
```sh
docker run -it yanncauchepin/lending_club:mlp_binary_classifier
```

If you would like to use a json data, please ensure you provide data in the same format as the *docker/data/data_test.json* file and well write the feature names. To bring the json data file in the docker image you must add a volume by running the following command. Assuming you have a folder named **data_folder** which contains your json data file *data.json*, enter the following command. Please adapt the *root_path* information according to your system.
```sh
docker run -it -v /root_path/data_folder:/app/data yanncauchepin/lendingclub:mlp_binary_classifier
```
By choosing this second choice of predicting the ***loan_status*** from a data json file, you can get the output by passing *data_folder/data.json* as input into the request. Nevertheless, if you inform *datat_test.json*, or simply press *Enter*, it will run the inner test.

---

![](featured_image.jpg)
