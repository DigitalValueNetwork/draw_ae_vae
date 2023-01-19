import {Tensor, SymbolicTensor, LayersModel} from "@tensorflow/tfjs-node"
// import {Layer}  from "@tensorflow/tfjs-layers/dist/engine"

export {Tensor, SymbolicTensor, LayersModel}

export const loadTfjsNode = () => import("@tensorflow/tfjs-node")

const loadTfjs = loadTfjsNode
export const loadTfjsGpu = () => import("@tensorflow/tfjs-node")

export type ITensorflow = Awaited<ReturnType<typeof loadTfjs>>
