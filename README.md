# rnn_test
Try to have an RNN emulate a simple sinus-function with random delayed outliers.


## Train and save

```bash
yarn run train
```

Output sample dataset:
```bash
yarn --silent run outputDataset > ./output/dataset.csv
```

Load existing model and output to csv:
```bash
rnn_test % node --inspect --loader ts-node/esm.mjs --experimental-json-modules src/main.ts --loadModelPath=file:///tmp/rnn_test/model.json > output/results.csv
```

(This will go straight to the prediction output)