import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { TricksWithModel } from './TricksWithModel';

const decoderUrl = "./decoderModel.json"

async function loadModel(modelUrl: string) {
  const decoder = await tf.loadLayersModel(modelUrl);

  const queryString = window.location.search.substring(1);
  if (queryString.match('debug')) {
    tfvis.show.modelSummary({name: 'decoder'}, decoder);
    tfvis.show.layer({name: 'dense2'}, decoder.getLayer('dense_Dense2'));
    tfvis.show.layer({name: 'dense3'}, decoder.getLayer('dense_Dense3'));
  }
  return decoder;
}


function App() {
  const [model, setModel] = useState<tf.LayersModel>()
  useEffect(() => {  
    loadModel(decoderUrl)
      .then(model => setModel(model))
      .catch(() => console.error("Error loading model"))
  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
        {
          !!model && <TricksWithModel  {...{model}} />
        }
      </header>
    </div>
  );
}

export default App;
