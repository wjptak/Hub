import hub

kaggle_credentials = {
    "username": "thisiseshan",
    "key": "c5a2a9fe75044da342e95a341f882f31",
}
ds = hub.Dataset.from_kaggle(
    tag="arturomoncadatorres/thanos-or-grimace",
    source="./datasets/thanosvgrimace/unstructured",
    destination="./datasets/thanosvgrimace/structured",
    kaggle_credentials=kaggle_credentials,
)
