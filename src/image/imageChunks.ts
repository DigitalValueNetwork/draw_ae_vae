import {asyncScheduler, bufferCount, flatMap, from, map, mergeMap, Observable, repeat, subscribeOn, take} from "rxjs"

export const shuffleIndices = (length: number) => {
	const indices = Array.from({ length }).map((_, i) => i)
	for (const { currentIndex, randomIndex } of Array.from({ length }).map((_, i) => ({ currentIndex: length - (i + 1), randomIndex: Math.floor(Math.random() * (length - i))}))) {
		[indices[randomIndex], indices[currentIndex]] = [indices[currentIndex], indices[randomIndex]]
	}
	return indices
}


export const imageChunks = (images: Float32Array[], batchSize: number, count = 2000) => {
	const arrayImages = images.map(image => [...image])
	return from([true]).pipe(
		mergeMap(() => from(shuffleIndices(arrayImages.length))),
		map((idx) => arrayImages[idx]),
		subscribeOn(asyncScheduler),
		repeat(),
		bufferCount(batchSize),
		take(Math.round(count / batchSize))
	)
}

export const imageChunkToFlat = (chunks: Observable<number[][]>) =>
	chunks.pipe(
		map(chunk => ({buffer: chunk.flat(), length: chunk.length})) // .reduce((x, y) => [...x, ...y])),
		// map(flatChunk => tf.tensor3d(flatChunk))
	)
