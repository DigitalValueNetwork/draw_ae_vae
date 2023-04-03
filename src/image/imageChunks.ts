import {asyncScheduler, bufferCount, from, map, Observable, repeat, subscribeOn} from "rxjs"

export const imageChunks = (images: Float32Array[], batchSize: number) => {
	return from(images).pipe(
		map(image => [...image]),
		subscribeOn(asyncScheduler),
		repeat(),
		bufferCount(batchSize)
	)
}

export const imageChunkToFlat = (chunks: Observable<number[][]>) =>
	chunks.pipe(
		map(chunk => ({buffer: chunk.flat(), length: chunk.length})) // .reduce((x, y) => [...x, ...y])),
		// map(flatChunk => tf.tensor3d(flatChunk))
	)
