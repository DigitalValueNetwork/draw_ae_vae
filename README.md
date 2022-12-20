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


## Lessons learned

* A huge issue in exploring AI is the efforts in developing the framework, and particularly the bugs
* The bug with deviating phases on the input and output caused the data to cancel itself out, leading to an average value being returned.
  * Don't think the signal spikes was present either, they were most likely covered in noise.
 

### From the data

<img width="1553" alt="image" src="https://user-images.githubusercontent.com/18142837/207315190-14923c09-4de1-46c4-ae4f-3c2d9388f343.png">

* The breaking point, when delta resets is not tranined - and creates some noise until it's past the lookback.
* After an event - there is significant noise for the lookback period.  Seems to hit partially on the output spike, but not always.
* Switching to GRU enabled a better response to the signal, when it was 3 steps away from the event.  It seemed to match the sinus curve better as well, but could just be more training.
* LSTM yielded more or less the same results, but with less matching sinus.


## Resources

https://stackoverflow.com/questions/13897316/approximating-the-sine-function-with-a-neural-network

