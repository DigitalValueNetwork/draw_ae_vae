import React, {useEffect, useRef, useState} from "react"

import * as tf from "@tensorflow/tfjs"

const IMAGE_HEIGHT = 28
const IMAGE_WIDTH = 28
const IMAGE_CHANNELS = 1
export const latentDim = 16

const decodeLatentSpaceTensor = (decoder: tf.LayersModel, inputTensor: tf.Tensor) => {
	return tf.tidy(() => {
		const res = (decoder.predict(inputTensor) as tf.Tensor).maximum(tf.scalar(0)).minimum(tf.scalar(1)) // .mul(255).cast("int32")
		const reshaped = res.reshape([inputTensor.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
		return reshaped
	})
}

export const GenerateImage = ({model, latentSpace}: {model: tf.LayersModel; latentSpace: number[]}) => {

	const [imageTensor, setImageTensor] = useState<tf.Tensor>()

	useEffect(() => {
		const tensor = decodeLatentSpaceTensor(model, tf.tensor(latentSpace, [1, latentDim]))
		setImageTensor(tensor)
	}, [latentSpace,  model])

	const canvasRef = useRef(null)

	useEffect(() => {
		if (imageTensor && canvasRef.current) {
			const myTensor = imageTensor.unstack()[0].resizeBilinear([IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2])
			tf.browser.toPixels(myTensor as any, canvasRef.current)	
		}
	}, [imageTensor])

	return (
		<div>
			<canvas width={50} height={50} ref={canvasRef} />
		</div>
	)
}
