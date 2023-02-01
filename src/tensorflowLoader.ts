// Huge issues with tensorflow typing when introducing tfjs-node-gpu
import {Tensor, SymbolicTensor, LayersModel, Shape} from "@tensorflow/tfjs-node"

// Exporting these types with the `type` keyword might make a difference in what content is actually imported above. 
export type {Tensor, SymbolicTensor, LayersModel, Shape}

export const loadTfjsNode = () => import("@tensorflow/tfjs-node")
export const loadTfjsGpu = () => import("@tensorflow/tfjs-node-gpu")

const loadTfjs = loadTfjsNode
export type ITensorflow = Awaited<ReturnType<typeof loadTfjs>>
