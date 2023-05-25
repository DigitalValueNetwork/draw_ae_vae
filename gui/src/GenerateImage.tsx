import React, {useEffect, useRef, useState} from "react"

import * as tf from "@tensorflow/tfjs"

const IMAGE_HEIGHT = 200
const IMAGE_WIDTH = 150
const IMAGE_CHANNELS = 3
export const latentDim = 5

/** Pass the latent space vector, in the form of a tensor, through the decoder - and generate the image */
const decodeLatentSpaceTensor = (decoder: tf.LayersModel, inputTensor: tf.Tensor) => {
	return tf.tidy(() => {
		const res = (decoder.predict(inputTensor) as tf.Tensor).maximum(tf.scalar(0)).minimum(tf.scalar(1))
		const reshaped = res.reshape([inputTensor.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
		return reshaped
	})
}

export const GenerateImage = ({model, latentSpace}: {model: tf.LayersModel; latentSpace: tf.Tensor}) => {
	const [imageTensor, setImageTensor] = useState<tf.Tensor>()

	useEffect(() => {
		const tensor = decodeLatentSpaceTensor(model, latentSpace)
		setImageTensor(tensor)
	}, [latentSpace, model])

	const canvasRef = useRef(null)

	useEffect(() => {
		tf.tidy(() => { 
			if (imageTensor && canvasRef.current) {
				const myTensor = imageTensor.unstack()[0].resizeBilinear([IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2])
				tf.browser.toPixels(myTensor as any, canvasRef.current)
			}	
		})
	}, [imageTensor])

	return (
		<div>
			<canvas width={IMAGE_WIDTH * 2} height={IMAGE_HEIGHT * 2} ref={canvasRef} />
		</div>
	)
}
