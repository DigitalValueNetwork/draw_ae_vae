# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Loading the model

Run the training in the main project, with `yarn train` then put the resulting stored model into: `./public` as `generatorModel.json` and `weights.bin`.

The RGB model should be stored as `generatorModel-rgb.json` and `weights-rgb.bin`.  NB: edit the json to get the correct link to the bin file. (rather: fix the output so it generates the right names).

```bash
cat output/decoder/model.json | python -m json.tool | sed 's/weights.bin/decoderWeights-rgb.bin/' > gui/public/decoderModel-rgb.json 
```

## Available Scripts

Check out the standard react-scripts.

