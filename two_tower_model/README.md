# Mission statement
I want to build a torch-based two tower model. The model will be used to do movie predictions as the dataset is good and it apparently is the toy dataset people are using. I did an admittedly rather brief search and did not find any better.

# Dataset
You can get your dataset here: https://grouplens.org/datasets/movielens/ . File paths assume that you download the 32M dataset.

Extract all files into the subfolder labelled data.

If you are not sure whether your machine can handle the load, you can limit the number of users and movies to be loaded. You will still load all the data at least once, but after that and most importantly before constructing any feature vectors, the data not needed will be discarded.

# Run the model
Like with the other demos here, I did not want to bloat my virtualenvs on my machine, so everything has to be invoked from the root folder of this project, e.g. with 
`pipenv run python -m two_tower_model.main`.

That also means that all paths in this little demo project are relative to the project's root folder.
One day, VSCode will collaborate with pipenv as nicely as Pycharm does, then I might(!) do a cleaner setup, but for now, this is good enough.