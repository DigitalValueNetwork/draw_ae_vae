import React from "react"

import * as tf from "@tensorflow/tfjs"

export const LatentVectorSliders = ({latentVector}: {latentVector: tf.Tensor}) => (
	<p style={{display: "flex", flexDirection: "column"}}>
		{((latentVector.arraySync() as number[][])[0] ?? []).map(x => (
			<input type="range" min="-2.5" max="2.5" step={0.001} value={x} />
		))}
	</p>
)
