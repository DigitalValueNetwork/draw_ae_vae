import React from "react"

import * as tf from "@tensorflow/tfjs"

export const LatentVectorSliders = ({latentVector, setLatentOverride}: {latentVector: tf.Tensor; setLatentOverride: (override: number[]) => void}) =>
	[
		{
			getLatentArray: () => (latentVector.arraySync() as number[][])[0],
		},
	]
		.map(({getLatentArray}) => ({
			getLatentArray,
			sendLatentUpdate: (idx: number, value: number) => setLatentOverride(getLatentArray().map((x, i) => (i === idx ? value : x))),
		}))
		.map(({getLatentArray, sendLatentUpdate}) => (
			<p style={{display: "flex", flexDirection: "column"}}>
				{(getLatentArray() ?? []).map((x, i) => (
					<input key={i} type="range" min="-2.5" max="2.5" step={0.001} value={x} onChange={({target}) => sendLatentUpdate(i, parseFloat(target.value))} />
				))}
			</p>
		))[0]
