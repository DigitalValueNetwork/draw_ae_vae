import terminalImage from 'terminal-image';
import { IImgProps, pixelArrayToPngByteBuffer } from './arrayToImage.js';

export const renderImageForTerminalPreview = (imageData: Float32Array, imgProps: IImgProps) =>
	pixelArrayToPngByteBuffer(imageData, imgProps)
		.then(pngBuffer => terminalImage.buffer(pngBuffer))
  