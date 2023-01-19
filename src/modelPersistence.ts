import tf from "@tensorflow/tfjs-node"

export const saveModel = (model: tf.LayersModel, path: string) =>
	model.save(path)
