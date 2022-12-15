import tf from "@tensorflow/tfjs-node-gpu"

export const saveModel = (model: tf.LayersModel, path: string) =>
	model.save(path)
