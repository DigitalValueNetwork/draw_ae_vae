import {Tensor, SymbolicTensor, LayersModel} from "@tensorflow/tfjs-node"
// import {Layer}  from "@tensorflow/tfjs-layers/dist/engine"

/* type Tensor = any
type SymbolicTensor = any
type LayersModel = any */

export type {Tensor, SymbolicTensor, LayersModel}

export const loadTfjsNode = () => import("@tensorflow/tfjs-node")

const loadTfjs = loadTfjsNode
export const loadTfjsGpu = () => import("@tensorflow/tfjs-node-gpu")

export type ITensorflow = Awaited<ReturnType<typeof loadTfjs>>
